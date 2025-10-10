"""
Compared with image_sample_sr3_1, supports image resolution of different width and length
"""

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append(".")
import torch as th
from guided_diffusion.image_datasets_seg_edge import load_val_data
from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from scripts.script_util import post_process
from skimage.metrics import peak_signal_noise_ratio
import pytorch_msssim


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
        in_channels=1,  # args.in_channels,
        random_crop=False,
        random_flip=False,
        deterministic=True,
        data_num=args.num_samples,
    )
    args.in_channels = args.in_channels // 2

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val")
    os.makedirs(result_path, exist_ok=True)
    for batch, cond in data:
        batch = batch.to(dist_util.dev())
        cond = {
            k: v.to(dist_util.dev())
            for k, v in cond.items()
        }
        batch_size = batch.shape[0]
        idx += 1
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
            avg_psnr += peak_signal_noise_ratio(_target_image.cpu().numpy(), _sample.cpu().numpy(), data_range=255)
            avg_ssim += pytorch_msssim.ssim(_target_image, _sample, data_range=255)
    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.log(f"avg_psnr:{avg_psnr:.4e}")
    logger.log(f"avg_ssim:{avg_ssim:.4e}")
    logger.logkv("avg_psnr", avg_psnr)
    logger.logkv("avg_ssim", avg_ssim)
    logger.dumpkvs()
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val_log_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val_log.txt")
    with open(val_log_path, 'a') as f:
        f.write(f" avg_psnr:{avg_psnr:.4e}")
        f.write(f" avg_ssim:{avg_ssim:.4e}")
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
