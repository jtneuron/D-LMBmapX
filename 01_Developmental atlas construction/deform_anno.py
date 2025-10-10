import argparse
import os
import sys
import numpy as np
import monai

from python_script.script_util import (
    add_dict_to_argparser,
    set_random_seed, args_to_dict,
)
from util import logger
from python_script.model_util import voxelmorph_defaults, create_model
import torch
import json
import time
from util.data_util.dataset import InferPairDataset
from model.registration.voxelmorph.voxelmorph import VxmDense
from util.metric.segmentation_metric import dice_metric3
from torch.utils.data import DataLoader
from model.registration.voxelmorph.layers import SpatialTransformer
from util.metric.registration_metric import folds_percent_metric, ssim_metric, jacobian_determinant, PSNR_metric, \
    NMAE_meric
from shutil import copyfile
from tqdm import tqdm
from util.visual.visual1.visual_image import cat_imgs, tensor2img, save_img
from util.visual.image_util import save_deformation_figure, save_det_figure, save_dvf_figure
import SimpleITK as sitk


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
        src_feats=2,
        trg_feats=2,
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


def main():
    set_random_seed(seed=0)
    args = create_argparser().parse_args()
    os.environ['LOGDIR'] = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    logger.configure()
    logger.log(list(vars(args).items()))
    logger.log(list(voxelmorph_defaults_var1().items()))
    logger.log(f"start test time: {time.asctime()}")
    test_path = os.path.abspath(__file__)
    logger.log(f"test file path: {test_path}")
    copyfile(test_path, os.path.join(logger.get_dir(), os.path.basename(test_path)))

    model_1 = create_model(**args_to_dict(args, voxelmorph_defaults_var1().keys()))
    model_2 = create_model(**args_to_dict(args, voxelmorph_defaults_var1().keys()))

    total_params = sum([param.nelement() for param in model_1.parameters()])
    logger.log("Number of parameter: %.2fM" % (total_params / 1e6))

    device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
    if len(args.checkpoint) > 0:
        logger.log(f"loading model from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model_1.load_state_dict(state_dict['model_1'])
        model_2.load_state_dict(state_dict['model_2'])
        copyfile(args.checkpoint, os.path.join(logger.get_dir(), os.path.basename(args.checkpoint)))

    model_1.to(device)
    model_2.to(device)
    gpu_num = len(args.gpu.split(","))

    if gpu_num > 1:
        model_1 = torch.nn.DataParallel(model_1, device_ids=[i for i in range(gpu_num)])
        model_2 = torch.nn.DataParallel(model_2, device_ids=[i for i in range(gpu_num)])

    # with open(args.data_dir, 'r') as f:
    #     dataset_config = json.load(f)

    transform = None

    num_classes = 6

    # test_dataset = InferPairDataset(dataset_config, dataset_type='test',
    #                                 registration_type=args.registration_type,
    #                                 transform=transform)
    # logger.log(f'test dataset size: {len(test_dataset)}')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    run_test(args, model_1, model_2, device, num_classes)
    logger.log(f"end test time: {time.asctime()}")


def extract_slices(img, i, batch_size, dim):
    if dim == 2:
        return img[:, :, i:i + batch_size, :, :]
    elif dim == 3:
        return img[:, :, :, i:i + batch_size, :]
    elif dim == 4:
        return img[:, :, :, :, i:i + batch_size]
    else:
        raise Exception("dim error")


def extract_reg_slices(volume1, volume2, label1, label2, anno, i, batch_size, dim):
    img1 = extract_slices(volume1, i, batch_size, dim)
    img2 = extract_slices(volume2, i, batch_size, dim)
    mask1 = extract_slices(label1, i, batch_size, dim) if label1 != [] else []
    mask2 = extract_slices(label2, i, batch_size, dim) if label2 != [] else []
    anno1 = extract_slices(anno, i, batch_size, dim)
    return img1, img2, mask1, mask2, anno1


def permute_img(img, B, C, H, W, D, dim):
    if dim == 2:
        return img.permute((0, 2, 1, 3, 4)).reshape((B * H, C, W, D)).contiguous()
    elif dim == 3:
        return img.permute((0, 3, 1, 2, 4)).reshape((B * W, C, H, D)).contiguous()
    elif dim == 4:
        return img.permute((0, 4, 1, 2, 3)).reshape((B * D, C, H, W)).contiguous()
    else:
        raise Exception("dim error")


def permute_reg_imgs(img1, img2, mask1, mask2, anno, B, C, H, W, D, dim):
    img1 = permute_img(img1, B, C, H, W, D, dim)
    img2 = permute_img(img2, B, C, H, W, D, dim)
    mask1 = permute_img(mask1, B, C, H, W, D, dim) if mask1 != [] else []
    mask2 = permute_img(mask2, B, C, H, W, D, dim) if mask2 != [] else []
    anno1 = permute_img(anno, B, C, H, W, D, dim)
    return img1, img2, mask1, mask2, anno1


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


def run_test(args, model_1, model_2, device, num_classes):
    model_1.eval()
    model_2.eval()
    batch_size = args.batch_size
    view_1_dim = args.view_1_dim
    view_2_dim = args.view_2_dim

    fixed = sitk.ReadImage(
        r"/media/user/phz/data/488/P28_ave_25um/P28.nii.gz")
    fixed = sitk.GetArrayFromImage(fixed)
    fixed = fixed[np.newaxis, np.newaxis, ...]
    fixed = torch.from_numpy(fixed).float()
    fixed = fixed / 255

    fixed_label = sitk.ReadImage(
        r"/media/user/phz/data/488/P28_ave_25um/P28_label.nii.gz")
    fixed_label = sitk.GetArrayFromImage(fixed_label)
    fixed_label = fixed_label[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label)

    moving = sitk.ReadImage(
        r"/media/user/phz/data/allen/allen_to_P28/reference_trans_affine_syn.nii.gz")
    moving = sitk.GetArrayFromImage(moving)
    moving = moving[np.newaxis, np.newaxis, ...]
    moving = torch.from_numpy(moving).float()
    moving = moving / 255

    moving_label = sitk.ReadImage(
        r"/media/user/phz/data/allen/allen_to_P28/reference_trans_affine_syn_label.nii.gz")
    moving_label = sitk.GetArrayFromImage(moving_label)
    moving_label = moving_label[np.newaxis, np.newaxis, ...]
    moving_label = torch.from_numpy(moving_label)

    moving_anno = sitk.ReadImage(
        r"/media/user/phz/data/allen/allen_to_P28/test/reference_trans_affine_syn_mask.nii.gz")
    moving_anno = sitk.GetArrayFromImage(moving_anno)
    # moving_anno = np.float(moving_anno)
    moving_anno = moving_anno[np.newaxis, np.newaxis, ...]

    moving_anno = torch.from_numpy(moving_anno)

    # allen的时候使用
    # moving_anno = torch.from_numpy(moving_anno.astype(int))

    with torch.no_grad():
        volume2, volume1 = fixed, moving
        label2, label1 = fixed_label, moving_label

        volume1 = volume1.to(device)
        label1 = label1.to(device)

        volume2 = volume2.to(device)
        label2 = label2.to(device)

        anno = moving_anno
        anno = anno.to(device)

        view_1_size = volume1.shape[view_1_dim]
        view_1_warp_volume1, view_1_warp_label1, view_1_dvf = [], [], []
        view_1_warp_label1_nearest, view_1_warp_anno1_nearest = [], []
        for i in range(0, view_1_size, batch_size):
            img1, img2, mask1, mask2, anno1 = extract_reg_slices(volume1, volume2, label1, label2,
                                                          anno, i, batch_size, view_1_dim)
            B, C, H, W, D = img1.shape
            img1, img2, mask1, mask2, anno1 = permute_reg_imgs(img1, img2, mask1, mask2, anno1, B, C, H, W, D, view_1_dim)

            warp_img1, warp_mask1, dvf_i = test_one_view(args, model_1, img1, img2, mask1, mask2, "val_view_1_")
            torch.cuda.empty_cache()

            if mask1 != []:
                warp_mask1_nearest = SpatialTransformer(mode='nearest')(mask1.float(), dvf_i)
                warp_mask1_nearest = unpermute_img(warp_mask1_nearest, B, C, H, W, D, view_1_dim)
                view_1_warp_label1_nearest.append(warp_mask1_nearest)

                warp_anno1_nearest = SpatialTransformer(mode='nearest')(anno1.float(), dvf_i)
                warp_anno1_nearest = unpermute_img(warp_anno1_nearest, B, C, H, W, D, view_1_dim)
                view_1_warp_anno1_nearest.append(warp_anno1_nearest)

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
        view_1_warp_anno1_nearest = torch.cat(view_1_warp_anno1_nearest, dim=view_1_dim)

        view_1_dvf = torch.cat(view_1_dvf, dim=view_1_dim)
        view_1_dvf = concat_dvf_3d(view_1_dvf, dim=view_1_dim)

        # warp_anno = SpatialTransformer(mode='nearest')(anno, view_1_dvf)

        # view_2_size = volume1.shape[view_2_dim]
        # view_2_warp_volume1, view_2_warp_label1, view_2_dvf = [], [], []
        # view_2_warp_label1_nearest = []
        # for i in range(0, view_2_size, batch_size):
        #     img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2,
        #                                                   i, batch_size, view_2_dim)
        #     B, C, H, W, D = img1.shape
        #     img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_2_dim)
        #
        #     warp_img1, warp_mask1, dvf_i = test_one_view(args, model_2, img1, img2, mask1, mask2, "val_view_2_")
        #     torch.cuda.empty_cache()
        #
        #     if mask1 != []:
        #         warp_mask1_nearest = SpatialTransformer(mode='nearest')(mask1.float(), dvf_i)
        #         warp_mask1_nearest = unpermute_img(warp_mask1_nearest, B, C, H, W, D, view_2_dim)
        #         view_2_warp_label1_nearest.append(warp_mask1_nearest)
        #
        #     warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_2_dim)
        #     warp_mask1 = unpermute_img(warp_mask1, B, C, H, W, D, view_2_dim)
        #     dvf_i = unpermute_img(dvf_i, B, dvf_i.shape[1], H, W, D, view_2_dim)
        #
        #     view_2_warp_volume1.append(warp_img1)
        #     if warp_mask1 != []:
        #         view_2_warp_label1.append(warp_mask1)
        #     view_2_dvf.append(dvf_i)
        #
        # view_2_warp_volume1 = torch.cat(view_2_warp_volume1, dim=view_2_dim)
        #
        # if view_2_warp_label1 != []:
        #     view_2_warp_label1 = torch.cat(view_2_warp_label1, dim=view_2_dim)
        # view_2_warp_label1_nearest = torch.cat(view_2_warp_label1_nearest, dim=view_2_dim)
        #
        # view_2_dvf = torch.cat(view_2_dvf, dim=view_2_dim)
        # view_2_dvf = concat_dvf_3d(view_2_dvf, dim=view_2_dim)

        # view_1_metric_dict = compute_metric(view_1_dvf, view_1_warp_volume1, view_1_warp_label1_nearest,
        #                                     volume2, label2, num_classes)
        # for key, value in view_1_metric_dict.items():
        #     logger.logkv_mean("val_view_1_" + key, value)

        # view_2_metric_dict = compute_metric(view_2_dvf, view_2_warp_volume1, view_2_warp_label1_nearest,
        #                                     volume2, label2, num_classes)
        # for key, value in view_2_metric_dict.items():
        #     logger.logkv_mean("val_view_2_" + key, value)

        if args.save_img:
            def write_image_3d(image, path, data_type):
                img = image.cpu().numpy().astype(data_type)
                img = sitk.GetImageFromArray(img)
                sitk.WriteImage(img, path)

            base_dir = os.path.join(logger.get_dir(), 'ave')
            os.makedirs(base_dir, exist_ok=True)
            # write_image_3d(ave_view1[0][0], os.path.join(base_dir, f"{'view1'}_ave_volume.nii.gz"), float)
            # write_image_3d(ave_view2[0][0], os.path.join(base_dir, f"{'view2'}_ave_volume.nii.gz"), float)
            # write_image_3d(ave_view1_label[0][0], os.path.join(base_dir, f"{'view1'}_ave_volume_label.nii.gz"), np.uint8)
            # write_image_3d(ave_view2_label[0][0], os.path.join(base_dir, f"{'view2'}_ave_volume_label.nii.gz"), np.uint8)

            # allen时为np.uint16
            write_image_3d(view_1_warp_volume1[0][0] * 255, os.path.join(base_dir, "488_deformed.nii.gz"),
                           np.uint8)
            write_image_3d(view_1_warp_anno1_nearest[0][0], os.path.join(base_dir, "anno_deformed.nii.gz"),
                           np.uint8)
            write_image_3d(view_1_warp_label1_nearest[0][0], os.path.join(base_dir, "label_deformed.nii.gz"),
                           np.uint8)

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


def test_one_view(args, model, img1, img2, mask1, mask2, identifier):
    _img1 = torch.cat([img1, mask1.float()], dim=1)
    _img2 = torch.cat([img2, mask2.float()], dim=1)
    dvf_i = model(_img1, _img2)
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf_i)
    warp_mask1 = []
    if mask1 != []:
        warp_mask1 = SpatialTransformer(mode='bilinear')(mask1.float(), dvf_i)
    return warp_img1, warp_mask1, dvf_i


def save_image_3d(dvf, warp_volume1, warp_label1, identifier, output_dir):
    def write_image_3d(image, path, data_type):
        img = image.cpu().numpy().astype(data_type)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, path)

    flow = dvf[0].permute(1, 2, 3, 0).contiguous()
    write_image_3d(flow, os.path.join(output_dir, f"{identifier}_flow.nii.gz"), float)
    write_image_3d(warp_volume1[0][0], os.path.join(output_dir, f"{identifier}_warp_volume.nii.gz"), float)
    write_image_3d(warp_label1[0][0], os.path.join(output_dir, f"{identifier}_warp_label.nii.gz"), np.uint8)


def display_metric(metric_dict, identifier):
    index_region = {1: 'hpf', 2: 'cp', 3: 'bs', 4: 'cbx', 5: 'ctx'}
    logger.logkv_mean(f"{identifier}_SSIM", metric_dict["SSIM"])
    logger.logkv_mean(f"{identifier}_folds_percent", metric_dict["folds_percent"])
    for i in range(metric_dict['dice'].shape[0]):
        logger.logkv_mean(f"{identifier}_dice_{index_region[i + 1]}", metric_dict['dice'][i].item())
    logger.logkv_mean(f"{identifier}_mean_dice", torch.mean(metric_dict['dice']).item())


def compute_metric(dvf, warp_volume1, warp_label1, volume2, label2, num_classes):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric3(warp_label1, label2, num_classes, reduction="none")

    # device = dvf.device
    # _warp_label1 = warp_label1.long().cpu()
    # _label2 = label2.long().cpu()
    #
    # metric_dict['ASD'] = ASD_metric3(_warp_label1, _label2, num_classes, reduction="none")
    # metric_dict['HD'] = HD_metric3(_warp_label1, _label2, num_classes, reduction="none")
    #
    # metric_dict['ASD'] = metric_dict['ASD'].to(device)
    # metric_dict['HD'] = metric_dict['HD'].to(device)

    metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2).item()
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()
    # metric_dict['folds_count'] = folds_count_metric(dvf).item()
    # metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict


def create_argparser():
    defaults = dict(
        view_1_dim=2,
        view_2_dim=4,
        checkpoint=r"/media/user/phz/registration_3/output/allen_to_P28/state000200.pt",
        batch_size=1,
        logdir=r"/media/user/phz/registration_3/output/allen_to_P28/test",
        gpu='0',
        registration_type=1,
        save_img=True
    )
    defaults.update(voxelmorph_defaults_var1())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == "__main__":
    main()
