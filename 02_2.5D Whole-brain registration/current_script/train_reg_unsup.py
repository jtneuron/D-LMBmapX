import argparse
import os
import sys

sys.path.append("/phz/registration_3")
from current_script.script_util import (
    add_dict_to_argparser,
    parse_resume_step_from_filename, set_random_seed, args_to_dict,
)
from util import logger
from current_script.model_util import voxelmorph_defaults
import torch
import json
import time
from util.data_util.dataset import TrainPairDataset, ValPairDataset
from torch.utils.data import DataLoader
from util.loss.NCCLoss import NCCLoss
from util.loss.MSELoss import MSELoss
from util.loss.GradientLoss2D import GradientLoss2D
from model.registration.voxelmorph.layers import SpatialTransformer
from util.metric.registration_metric import folds_percent_metric, ssim_metric, jacobian_determinant, PSNR_metric, \
    NMAE_meric
from util.ModelSaver import ModelSaver
from shutil import copyfile
from tqdm import tqdm
from torch.autograd import Variable
# from util.visual.visual1.visual_image import cat_imgs, tensor2img, save_img
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
        src_feats=1,
        trg_feats=1,
    )
    return config


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
    logger.log(list(voxelmorph_defaults().items()))
    logger.log(f"start train time: {time.asctime()}")
    train_path = os.path.abspath(__file__)
    logger.log(f"train file path: {train_path}")
    copyfile(train_path, os.path.join(logger.get_dir(), os.path.basename(train_path)))

    model_1 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))
    model_2 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))

    total_params = sum([param.nelement() for param in model_1.parameters()])
    logger.log("Number of parameter: %.2fM" % (total_params / 1e6))

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=args.lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=args.lr)
    device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
    resume_epoch = 0
    if len(args.resume_checkpoint) > 0:
        logger.log(f"loading model from {args.resume_checkpoint}")
        resume_epoch = parse_resume_step_from_filename(args.resume_checkpoint)
        state_dict = torch.load(args.resume_checkpoint, map_location=device)
        model_1.load_state_dict(state_dict['model_1'])
        optimizer_1.load_state_dict(state_dict['optimizer_1'])

        for state in optimizer_1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        model_2.load_state_dict(state_dict['model_2'])
        optimizer_2.load_state_dict(state_dict['optimizer_2'])

        for state in optimizer_2.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    model_1.to(device)
    model_2.to(device)
    gpu_num = len(args.gpu.split(","))

    if gpu_num > 1:
        model_1 = torch.nn.DataParallel(model_1, device_ids=[i for i in range(gpu_num)])
        model_2 = torch.nn.DataParallel(model_2, device_ids=[i for i in range(gpu_num)])
    model_without_ddp_1 = model_1.module if gpu_num > 1 else model_1
    model_without_ddp_2 = model_2.module if gpu_num > 1 else model_2

    if args.data_dir.endswith('toml'):
        with open(args.data_dir, 'r') as f:
            dataset_config = toml.load(f)
    elif args.data_dir.endswith('json'):
        with open(args.data_dir, 'r') as f:
            dataset_config = json.load(f)
    else:
        raise Exception("dataset config file format error")

    registration_type = dataset_config["registration_type"]

    transform = None
    if len(args.resolution) > 0:
        transform = monai.transforms.ResizeD(spatial_size=args.resolution,
                                             keys=['volume1', 'volume2'],
                                             mode=['trilinear', 'trilinear'])

        # transform = monai.transforms.ResizeD(spatial_size=args.resolution,
        #                                      keys=['volume1', 'volume2', 'label1', 'label2'],
        #                                      mode=['trilinear', 'trilinear', 'nearest', 'nearest'])

    train_dataset = TrainPairDataset(dataset_config, dataset_type='train',
                                     registration_type=registration_type,
                                     transform=transform)

    # val_dataset = ValPairDataset(dataset_config, dataset_type='test',
    #                              registration_type=registration_type,
    #                              transform=transform)
    logger.log(f'train dataset size: {len(train_dataset)}')
    # logger.log(f'val dataset size: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    modelsaver = ModelSaver(args.max_save_num)
    for epoch in range(resume_epoch + 1, args.epoches + 1):
        run_train(args, model_1, model_2, optimizer_1, optimizer_2, train_dataloader, device, epoch)
        # if epoch % args.val_interval == 0:
        #     run_val(args, model_1, model_2, train_dataloader, device, epoch)
        if epoch % args.save_interval == 0:
            state_dict = {"model_1": model_without_ddp_1.state_dict(),
                          "model_2": model_without_ddp_2.state_dict(),
                          "optimizer_1": optimizer_1.state_dict(),
                          "optimizer_2": optimizer_2.state_dict()}
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


def extract_reg_slices(volume1, volume2, i, batch_size, dim):
    img1 = extract_slices(volume1, i, batch_size, dim)
    img2 = extract_slices(volume2, i, batch_size, dim)
    return img1, img2


def permute_img(img, B, C, H, W, D, dim):
    if dim == 2:
        return img.permute((0, 2, 1, 3, 4)).reshape((B * H, C, W, D)).contiguous()
    elif dim == 3:
        return img.permute((0, 3, 1, 2, 4)).reshape((B * W, C, H, D)).contiguous()
    elif dim == 4:
        return img.permute((0, 4, 1, 2, 3)).reshape((B * D, C, H, W)).contiguous()
    else:
        raise Exception("dim error")


def permute_reg_imgs(img1, img2, B, C, H, W, D, dim):
    img1 = permute_img(img1, B, C, H, W, D, dim)
    img2 = permute_img(img2, B, C, H, W, D, dim)
    return img1, img2


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


def run_train(args, model_1, model_2, optimizer_1, optimizer_2, train_dataloader, device, epoch):
    model_1.train()
    model_2.train()
    batch_size = args.batch_size
    view_1_dim = args.view_1_dim
    view_2_dim = args.view_2_dim
    logger.logkv("epoch", epoch)
    for img_dict in tqdm(train_dataloader):
        volume1, volume2 = img_dict['volume1'], img_dict['volume2']
        volume1 = volume1.to(device)
        volume2 = volume2.to(device)

        view_1_size = volume1.shape[view_1_dim]
        view_1_warp_volume1 = []

        for i in range(0, view_1_size, batch_size):
            img1, img2 = extract_reg_slices(volume1, volume2, i, batch_size, view_1_dim)
            B, C, H, W, D = img1.shape
            img1, img2 = permute_reg_imgs(img1, img2, B, C, H, W, D, view_1_dim)

            warp_img1 = train_one_view(args, model_1, optimizer_1, img1, img2, "view_1_")

            warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_1_dim)
            view_1_warp_volume1.append(warp_img1)

        view_1_warp_volume1 = torch.cat(view_1_warp_volume1, dim=view_1_dim)

        view_2_size = volume1.shape[view_2_dim]
        view_2_warp_volume1 = []
        for i in range(0, view_2_size, batch_size):
            img1, img2 = extract_reg_slices(volume1, volume2, i, batch_size, view_2_dim)
            B, C, H, W, D = img1.shape
            img1, img2 = permute_reg_imgs(img1, img2, B, C, H, W, D, view_2_dim)

            warp_img1 = train_one_view(args, model_2, optimizer_2, img1, img2, "view_2_")

            warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_2_dim)

            view_2_warp_volume1.append(warp_img1)

        view_2_warp_volume1 = torch.cat(view_2_warp_volume1, dim=view_2_dim)

        loss_dict = compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss_dict["consistency_total_loss"].backward()
        optimizer_1.step()
        optimizer_2.step()
        for key, value in loss_dict.items():
            logger.logkv_mean(key, value.item())

    logger.dumpkvs()


def train_one_view(args, model, optimizer, img1, img2, identifier):
    dvf_i = model(img1, img2)
    loss_dict = compute_loss(args, dvf_i, img1, img2)
    for key, value in loss_dict.items():
        logger.logkv_mean(identifier + key, value.item())
    optimizer.zero_grad()
    loss_dict["total_loss"].backward()
    optimizer.step()
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf_i)
    return warp_img1


def run_val(args, model_1, model_2, val_dataloader, device, epoch):
    model_1.eval()
    model_2.eval()
    batch_size = args.batch_size
    view_1_dim = args.view_1_dim
    view_2_dim = args.view_2_dim
    with torch.no_grad():
        for img_dict in tqdm(val_dataloader):

            volume1, volume2 = img_dict['volume1'], img_dict['volume2']
            id1, id2 = img_dict['id1'][0], img_dict['id2'][0]

            volume1 = volume1.to(device)
            volume2 = volume2.to(device)

            view_1_size = volume1.shape[view_1_dim]
            view_1_warp_volume1, view_1_dvf = [], []
            for i in range(0, view_1_size, batch_size):
                img1, img2 = extract_reg_slices(volume1, volume2, i, batch_size, view_1_dim)
                B, C, H, W, D = img1.shape
                img1, img2 = permute_reg_imgs(img1, img2, B, C, H, W, D, view_1_dim)
                warp_img1, dvf_i = val_one_view(args, model_1, img1, img2, "val_view_1_")

                warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_1_dim)

                dvf_i = unpermute_img(dvf_i, B, dvf_i.shape[1], H, W, D, view_1_dim)

                view_1_warp_volume1.append(warp_img1)
                view_1_dvf.append(dvf_i)

            view_1_warp_volume1 = torch.cat(view_1_warp_volume1, dim=view_1_dim)

            view_1_dvf = torch.cat(view_1_dvf, dim=view_1_dim)
            view_1_dvf = concat_dvf_3d(view_1_dvf, view_1_dim)

            view_2_size = volume1.shape[view_2_dim]
            view_2_warp_volume1, view_2_dvf = [], []
            for i in range(0, view_2_size, batch_size):
                img1, img2 = extract_reg_slices(volume1, volume2, i, batch_size, view_2_dim)
                B, C, H, W, D = img1.shape
                img1, img2 = permute_reg_imgs(img1, img2, B, C, H, W, D, view_2_dim)
                warp_img1, dvf_i = val_one_view(args, model_2, img1, img2, "val_view_2_")

                warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_2_dim)

                dvf_i = unpermute_img(dvf_i, B, dvf_i.shape[1], H, W, D, view_2_dim)
                view_2_warp_volume1.append(warp_img1)
                view_2_dvf.append(dvf_i)

            view_2_warp_volume1 = torch.cat(view_2_warp_volume1, dim=view_2_dim)

            view_2_dvf = torch.cat(view_2_dvf, dim=view_2_dim)
            view_2_dvf = concat_dvf_3d(view_2_dvf, dim=view_2_dim)

            loss_dict = compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1)
            for key, value in loss_dict.items():
                logger.logkv_mean("val_" + key, value.item())

            view_1_metric_dict = compute_metric(view_1_dvf, view_1_warp_volume1, volume2)
            for key, value in view_1_metric_dict.items():
                logger.logkv_mean("val_view_1_" + key, value)

            view_2_metric_dict = compute_metric(view_2_dvf, view_2_warp_volume1, volume2)
            for key, value in view_2_metric_dict.items():
                logger.logkv_mean("val_view_2_" + key, value)
            output_dir = os.path.join(logger.get_dir(), "val_img", str(epoch), id1 + "_" + id2)
            os.makedirs(output_dir, exist_ok=True)

            save_img_list(img_list1=[volume1, volume2, view_1_warp_volume1],
                          view_dim=view_1_dim,
                          output_path=os.path.join(output_dir, "view_1.png")
                          )
            save_img_list(img_list1=[volume1, volume2, view_2_warp_volume1],
                          view_dim=view_2_dim,
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


def save_img_list(img_list1, output_path, view_dim):
    interval = img_list1[0].shape[view_dim] // 15
    img_list1 = [img.cpu() for img in img_list1]
    img = cat_imgs(img_list1, interval=interval, view_dim=view_dim)
    img = tensor2img(img, nrow=1, min_max=(0, 1))
    save_img(img, output_path)


def val_one_view(args, model, img1, img2, identifier):
    dvf_i = model(img1, img2)

    loss_dict = compute_loss(args, dvf_i, img1, img2)
    for key, value in loss_dict.items():
        logger.logkv_mean(identifier + key, value.item())
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf_i)

    return warp_img1, dvf_i


def compute_metric(dvf, warp_volume1, volume2):
    metric_dict = dict()
    # metric_dict['dice'] = dice_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['ASD'] = ASD_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['HD'] = HD_metric2(warp_label1, label2, num_classes).item()
    metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2).item()
    metric_dict['PSNR'] = PSNR_metric(warp_volume1, volume2, data_range=1).item()
    metric_dict['NMAE'] = NMAE_meric(warp_volume1, volume2).item()
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()

    # metric_dict['folds_count'] = folds_count_metric(dvf).item()
    # metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict


def compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1):
    # similarity_loss = NCCLoss(spatial_dims=3, kernel_size=9)(view_1_warp_volume1, view_2_warp_volume1)

    total_loss = MSELoss()(view_1_warp_volume1, view_2_warp_volume1)
    total_loss = Variable(total_loss, requires_grad=True)
    loss_dict = {"consistency_total_loss": total_loss}

    return loss_dict


def compute_loss(args, dvf, img1, img2):
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf)
    similarity_loss = NCCLoss(spatial_dims=2, kernel_size=9)(warp_img1, img2)
    smooth_loss = GradientLoss2D()(dvf)
    total_loss = similarity_loss + args.smooth_loss_weight * smooth_loss
    loss_dict = {"total_loss": total_loss, "similarity_loss": similarity_loss,
                 "smooth_loss": smooth_loss}
    return loss_dict


def create_argparser():
    defaults = dict(
        view_1_dim=2,
        view_2_dim=4,
        data_dir="/mnt/18TB_HDD2/phz/cross_stage/config/dataset_config/P4_488_to_4.toml",
        lr=1e-4,
        batch_size=4,
        log_interval=10,
        save_interval=50,
        resume_checkpoint="",
        logdir="/mnt/18TB_HDD2/phz/reg/registration_3/output/P4_488_to_4_unsup",
        gpu='1',
        max_save_num=30,
        val_interval=10,
        epoches=200,
        registration_type=4,
        seg_loss_weight=1.,
        smooth_loss_weight=1.,
        resolution="",
    )
    defaults.update(voxelmorph_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
