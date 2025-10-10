"""
Compared with image_sample_sr3_1, supports image resolution of different width and length
"""

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import os.path
import sys

sys.path.append(".")
import torch as th
import torch
from guided_diffusion.image_datasets_pair_target_1 import load_val_data
from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from scripts.script_util import post_process,save_img
from skimage.metrics import peak_signal_noise_ratio
import pytorch_msssim
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
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
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

    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val")
    os.makedirs(result_path, exist_ok=True)
    
    img_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "img")
    os.makedirs(img_path, exist_ok=True)
    
    for idx, (batch, cond) in enumerate(data):
        batch = batch.to(dist_util.dev())
        cond = {
            k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
            for k, v in cond.items() 
        }
        batch_size = batch.shape[0]
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            tuple(batch.shape),
            clip_denoised=True,
            model_kwargs=cond,
            progress=True,
            continous=args.continous,
        )
        sample = post_process(sample, batch)
        
        for i in range(batch.shape[0]):
            save_img(sample[i][0], img_path, cond['image_path'][i], 'fake_B')
            save_img(batch[i][0], img_path, cond['image_path'][i], 'real_A')
            if cond.get("target_image", None) is not None: 
                save_img(cond["target_image"][i][0], img_path, cond['image_path'][i], 'real_B')
        
        if args.continous:
            continous_img = sample[1]
            sample = sample[0]
            continous_img = metrics.tensor2img(continous_img, nrow=batch_size)
            metrics.save_img(continous_img, '{}/{}_continous_img.png'.format(result_path, idx))
        if cond.get("target_image", None) is not None:
            # concat_imgs = th.cat((cond["cond_image"], batch, cond["target_image"], sample), dim=-1)
            concat_imgs = th.cat((cond["cond_image"], batch, sample), dim=-1)
            targets = metrics.tensor2img(cond["target_image"])
        else:
            concat_imgs = th.cat((cond["cond_image"], batch, sample), dim=-1)
        concat_imgs = metrics.tensor2img(concat_imgs, nrow=2)
        cond_images = metrics.tensor2img(cond["cond_image"])
        sources = metrics.tensor2img(batch)
        translated = metrics.tensor2img(sample)
        metrics.save_img(concat_imgs, '{}/{}_concatenate.png'.format(result_path, idx))
        metrics.save_img(cond_images, '{}/{}_cond_images.png'.format(result_path, idx))
        metrics.save_img(sources, '{}/{}_sources.png'.format(result_path, idx))
        metrics.save_img(translated, '{}/{}_translated.png'.format(result_path, idx))
        if cond.get("target_image", None) is not None:
            metrics.save_img(targets, '{}/{}_targets.png'.format(result_path, idx))
            _target_image = ((cond["target_image"] + 1) * 127.5)
            _sample = ((sample + 1) * 127.5)
            for i in range(_target_image.shape[0]):
                _target_image_i = _target_image[i][None]
                _sample_i = _sample[i][None]
                if torch.count_nonzero(_target_image_i).item() > 0:
                    _psnr = peak_signal_noise_ratio(_target_image_i.cpu().numpy(), _sample_i.cpu().numpy(),
                                                    data_range=255)
                    _ssim = pytorch_msssim.ssim(_target_image_i, _sample_i, data_range=255)
                    _mae = mean_absolute_error(_target_image_i.cpu().numpy(), _sample_i.cpu().numpy(), data_range=255)
                    psnr_list.append(_psnr.item())
                    ssim_list.append(_ssim.cpu().item())
                    mae_list.append(_mae.item())
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mae = np.mean(mae_list)
    
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_mae = np.std(mae_list)
    
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.log(f"avg_psnr:{avg_psnr:.4e}")
    logger.log(f"avg_ssim:{avg_ssim:.4e}")
    logger.log(f"avg_mae: {avg_mae:.4e}")
    
    logger.log(f"std_psnr:{std_psnr:.4e}")
    logger.log(f"std_ssim:{std_ssim:.4e}")
    logger.log(f"std_mae: {std_mae:.4e}")
    
    logger.logkv("avg_psnr", avg_psnr)
    logger.logkv("avg_ssim", avg_ssim)
    logger.logkv("avg_mae", avg_mae)

    logger.logkv("std_psnr", std_psnr)
    logger.logkv("std_ssim", std_ssim)
    logger.logkv("std_mae", std_mae)

    logger.dumpkvs()
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val_log_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val_log.txt")
    with open(val_log_path, 'a') as f:
        f.write(f"avg_psnr:{avg_psnr:.4e}\n")
        f.write(f"avg_ssim:{avg_ssim:.4e}\n")
        f.write(f"avg_mae:{avg_mae:.4e}\n")

        f.write(f"std_psnr:{std_psnr:.4e}\n")
        f.write(f"std_ssim:{std_ssim:.4e}\n")
        f.write(f"std_mae:{std_mae:.4e}\n")

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
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == "__main__":
    main()
