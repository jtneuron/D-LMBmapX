"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys

sys.path.append(".")

from scripts.script_util import get_cond, get_dice_and_mask, post_process,save_img
from scripts.common import read_model_and_diffusion_1
import torch
from guided_diffusion.image_datasets_pair_1 import load_val_data
from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from skimage.metrics import peak_signal_noise_ratio
import pytorch_msssim
import torch as th
import numpy as np

def mean_absolute_error(fake, real,data_range=1.):
    b,c,x, y = np.where(real != -1)  # Exclude background
    mae = np.abs(fake[b,c,x, y] - real[b,c,x, y]).mean()
    return mae / data_range

def main():
    args = create_argparser().parse_args()
    args.resolution = tuple(args.resolution)
    os.environ['OPENAI_LOGDIR'] = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    source_model, diffusion = read_model_and_diffusion_1(args, args.source_path)
    target_model, _ = read_model_and_diffusion_1(args, args.target_path)

    logger.log(f"source_seg_model_path: {args.source_seg_model_path}")
    logger.log(f"target_seg_model_path: {args.target_seg_model_path}")

    source_seg_model = torch.load(args.source_seg_model_path)
    source_seg_model.to(dist_util.dev())
    source_seg_model.eval()
    source_num_classes = source_seg_model.out_channels

    target_seg_model = torch.load(args.target_seg_model_path)
    target_seg_model.to(dist_util.dev())
    target_seg_model.eval()
    target_num_classes = target_seg_model.out_channels
    assert source_num_classes == target_num_classes
    logger.log(args)
    logger.log(args_to_dict(args, model_and_diffusion_defaults().keys()))
    logger.log("sampling...")

    data = load_val_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.resolution,
        class_cond=args.class_cond,
        in_channels=args.in_channels,
        random_crop=False,
        random_flip=False,
        deterministic=True,
        data_num=args.num_samples,
    )
    args.in_channels = args.in_channels // 2

    psnr_list = []
    ssim_list = []
    mae_list = []
    source_val_dice = [[] for _ in range(target_num_classes-1)]
    target_val_dice = [[] for _ in range(target_num_classes-1)]
    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val")
    os.makedirs(result_path, exist_ok=True)
    
    img_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "img")
    os.makedirs(img_path, exist_ok=True)
    cnt=0
    for idx, (batch, cond) in enumerate(data):
        batch = batch.to(dist_util.dev())
        cond = {
            k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
            for k, v in cond.items()
        }
        sample_fn = diffusion.p_sample_loop
        sample_target = sample_fn(
            target_model,
            tuple(batch.shape),
            clip_denoised=True,
            model_kwargs=cond,
            progress=True,
        )
        sample_target = post_process(sample_target, batch)
        target_cond = get_cond(sample_target.cpu().numpy())
        target_cond = {
            k: v.to(dist_util.dev())
            for k, v in target_cond.items()
        }

        sample_source = sample_fn(
            source_model,
            tuple(sample_target.shape),
            clip_denoised=True,
            model_kwargs=target_cond,
            progress=True,
        )

        sample_source = post_process(sample_source, batch)
        
        for i in range(batch.shape[0]):
            cnt+=1
            save_img(sample_target[i][0], img_path, cond['image_path'][i], 'fake_B')
            save_img(batch[i][0], img_path, cond['image_path'][i], 'real_A')
            save_img(sample_source[i][0], img_path, cond['image_path'][i], 'recovered_A')
            if cond.get("target_image", None) is not None: 
                save_img(cond["target_image"][i][0], img_path, cond['image_path'][i], 'real_B')
        
        if cond.get("target_image", None) is not None:
            concat_imgs = torch.cat((batch, sample_target, sample_source), dim=-1)
            targets = metrics.tensor2img(cond["target_image"])
        else:
            concat_imgs = torch.cat((batch, sample_target, sample_source), dim=-1)

        # concat_imgs = metrics.tensor2img(concat_imgs, nrow=2)
        cond_images = metrics.tensor2img(cond["cond_image"])
        sources = metrics.tensor2img(batch)
        sample_target_imgs = metrics.tensor2img(sample_target)
        sample_source_imgs = metrics.tensor2img(sample_source)
        target_cond_imgs = metrics.tensor2img(target_cond["cond_image"])
        # metrics.save_img(concat_imgs, '{}/{}_concatenate.png'.format(result_path, idx))
        metrics.save_img(cond_images, '{}/{}_cond_images.png'.format(result_path, idx))
        metrics.save_img(sources, '{}/{}_sources.png'.format(result_path, idx))
        metrics.save_img(target_cond_imgs, '{}/{}_target_cond.png'.format(result_path, idx))
        metrics.save_img(sample_target_imgs, '{}/{}_sample_target.png'.format(result_path, idx))
        metrics.save_img(sample_source_imgs, '{}/{}_sample_source.png'.format(result_path, idx))
        if cond.get("target_image", None) is not None:
            metrics.save_img(targets, '{}/{}_targets.png'.format(result_path, idx))
        _groud_true = ((batch + 1) * 127.5)
        _predict = ((sample_source + 1) * 127.5)
        for i in range(_groud_true.shape[0]):
            _groud_true_i = _groud_true[i][None]
            _predict_i = _predict[i][None]
            if torch.count_nonzero(_groud_true_i).item() > 0:
                _psnr = peak_signal_noise_ratio(_groud_true_i.cpu().numpy(), _predict_i.cpu().numpy(), data_range=255)
                _ssim = pytorch_msssim.ssim(_groud_true_i, _predict_i, data_range=255)
                _mae = mean_absolute_error(_groud_true_i.cpu().numpy(), _predict_i.cpu().numpy(), data_range=255)
                psnr_list.append(_psnr.item())
                ssim_list.append(_ssim.cpu().item())
                mae_list.append(_mae.item())

        with torch.no_grad():
            labels = cond["mask_image"]
            t = th.zeros(sample_source.shape[0], dtype=th.long, device=dist_util.dev())
            predict = source_seg_model(sample_source, timesteps=t)
            source_dice, predict_source_mask = get_dice_and_mask(labels, predict, source_num_classes)

            # source_val_dice.append(_dice.cpu().numpy().tolist())

            t = th.zeros(sample_target.shape[0], dtype=th.long, device=dist_util.dev())
            predict = target_seg_model(sample_target, timesteps=t)
            target_dice, predict_target_mask = get_dice_and_mask(labels, predict, target_num_classes)

            # target_val_dice.append(_dice.cpu().numpy().tolist())

            _labels = (labels / source_num_classes) * 2. - 1.

            concat_imgs = th.cat((concat_imgs, _labels, predict_target_mask, predict_source_mask), dim=-1)
            concat_imgs = metrics.tensor2img(concat_imgs, nrow=1)
            metrics.save_img(concat_imgs, '{}/{}_concat.png'.format(result_path, idx))
            for mask_id in torch.unique(labels).cpu().numpy().tolist():
                if mask_id:
                    source_val_dice[mask_id-1].append(source_dice[mask_id - 1].cpu().item())
                    target_val_dice[mask_id-1].append(target_dice[mask_id - 1].cpu().item())
    
    
    logger.log(str(source_val_dice))
    logger.log(str(target_val_dice))
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mae = np.mean(mae_list)

    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_mae = np.std(mae_list)

    logger.logkv("avg_psnr", avg_psnr)
    logger.logkv("avg_ssim", avg_ssim)
    logger.logkv("avg_mae", avg_mae)

    logger.logkv("std_psnr", std_psnr)
    logger.logkv("std_ssim", std_ssim)
    logger.logkv("std_mae", std_mae)
    logger.dumpkvs()

    index_region = {1: 'hpf', 2: 'cp', 3: 'bs', 4: 'cbx', 5: 'ctx'}

    mean_source_dice, std_source_dice = [], []
    for i in range(len(source_val_dice)):
        mean_source_dice.append(np.mean(source_val_dice[i]) if len(source_val_dice[i]) > 0 else 0.)
        std_source_dice.append(np.std(source_val_dice[i]) if len(source_val_dice[i]) > 0 else 0.)

    mean_target_dice, std_target_dice = [], []
    for i in range(len(target_val_dice)):
        mean_target_dice.append(np.mean(target_val_dice[i]) if len(target_val_dice[i]) > 0 else 0.)
        std_target_dice.append(np.std(target_val_dice[i]) if len(target_val_dice[i]) > 0 else 0.)

    avg_source_dice = np.mean(mean_source_dice)
    avg_target_dice = np.mean(mean_target_dice)
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    logger.log("+++++++++++++++++++ target +++++++++++++++++++++++++++++")
    logger.logkv("avg_dice", avg_target_dice)

    for i in range(len(mean_target_dice)):
        logger.logkv(f"{index_region[i + 1]}_dice", mean_target_dice[i])
    logger.dumpkvs()

    logger.log("+++++++++++++++++++ source +++++++++++++++++++++++++++++")
    logger.logkv("avg_dice", avg_source_dice)
    for i in range(len(mean_source_dice)):
        logger.logkv(f"{index_region[i + 1]}_dice", mean_source_dice[i])
    logger.dumpkvs()
    
    val_log_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val_log.txt")
    with open(val_log_path, 'a') as f:
        f.write(f" avg_psnr:{avg_psnr}\n")
        f.write(f" avg_ssim:{avg_ssim}\n")
        f.write(f" avg_mae:{avg_mae}\n")

        f.write(f" std_psnr:{std_psnr}\n")
        f.write(f" std_ssim:{std_ssim}\n")
        f.write(f" std_mae:{std_mae}\n")
        
        f.write("++++++++++++++ target ++++++++++++++++\n")

        f.write(f"avg_dice: {avg_target_dice}\n")
        for i in range(len(mean_target_dice)):
            f.write(f"{index_region[i + 1]}_mean_dice:{mean_target_dice[i]}  std_dice: {std_target_dice[i]}\n")

        f.write("\n\n++++++++++++++ source ++++++++++++++++\n")
        f.write(f"avg_dice: {avg_source_dice}\n")

        for i in range(len(mean_source_dice)):
            f.write(f"{index_region[i + 1]}_mean_dice:{mean_source_dice[i]} std_dice: {std_source_dice[i]}\n")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        logdir="./output/temp",
        gpu='0',
        data_dir='',
        continous=False,
        source_path='',
        target_path='',
        source_seg_model_path='',
        target_seg_model_path='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == "__main__":
    main()
