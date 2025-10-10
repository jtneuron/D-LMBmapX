"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import sys

sys.path.append(".")

from scripts.script_util import get_cond, post_process

from scripts.common import read_model_and_diffusion_segbranch
import torch
from guided_diffusion.image_datasets_pair import load_val_data
from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
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
    source_model, diffusion = read_model_and_diffusion_segbranch(args, args.source_path)
    target_model, _ = read_model_and_diffusion_segbranch(args, args.target_path)

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
        isTrain=False,
        type=args.val_type,
    )
    args.in_channels = args.in_channels // 2

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val")
    os.makedirs(result_path, exist_ok=True)
    for batch, cond, paths in data:
        batch = batch.to(dist_util.dev())
        cond = {
            k: v.to(dist_util.dev())
            for k, v in cond.items()
        }
        idx += 1
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
            model_kwargs=target_cond,  # target_cond,cond
            progress=True,
        )

        sample_source = post_process(sample_source, batch)

        if cond.get("target_image", None) is not None:
            concat_imgs = torch.cat((batch, sample_target, sample_source), dim=-1)
            # concat_imgs = torch.cat((cond["cond_image"], target_cond["cond_image"], batch,
            #                               cond["target_image"], sample_target, sample_source), dim=-1)
            targets = metrics.tensor2img(cond["target_image"])
        else:
            # concat_imgs = torch.cat((cond["cond_image"], target_cond["cond_image"], batch,
            #                               sample_target, sample_source), dim=-1)
            concat_imgs = torch.cat((batch, sample_target, sample_source), dim=-1)
        concat_imgs = metrics.tensor2img(concat_imgs, nrow=2)
        cond_images = metrics.tensor2img(cond["cond_image"])
        sources = metrics.tensor2img(batch)
        sample_target_imgs = metrics.tensor2img(sample_target)
        sample_source_imgs = metrics.tensor2img(sample_source)
        target_cond_imgs = metrics.tensor2img(target_cond["cond_image"])
        metrics.save_img(concat_imgs, '{}/{}_concatenate.png'.format(result_path, idx))
        # metrics.save_img(cond_images, '{}/{}_cond_images.png'.format(result_path, idx))
        # metrics.save_img(sources, '{}/{}_sources.png'.format(result_path, idx))
        # metrics.save_img(target_cond_imgs, '{}/{}_target_cond.png'.format(result_path, idx))
        # metrics.save_img(sample_target_imgs, '{}/{}_sample_target.png'.format(result_path, idx))
        # metrics.save_img(sample_source_imgs, '{}/{}_sample_source.png'.format(result_path, idx))
        metrics.save_img_batch(cond_images, result_path, paths, "cond_images")
        metrics.save_img_batch(sources, result_path, paths, "sources")
        metrics.save_img_batch(target_cond_imgs, result_path, paths, "target_cond")
        metrics.save_img_batch(sample_target_imgs, result_path, paths, "sample_target")
        metrics.save_img_batch(sample_source_imgs, result_path, paths, "sample_source")
        if cond.get("target_image", None) is not None:
            metrics.save_img(targets, '{}/{}_targets.png'.format(result_path, idx))
        _groud_true = ((batch + 1) * 127.5)
        _predict = ((sample_source + 1) * 127.5)
        avg_psnr += peak_signal_noise_ratio(_groud_true.cpu().numpy(), _predict.cpu().numpy(), data_range=255)
        avg_ssim += pytorch_msssim.ssim(_groud_true, _predict, data_range=255)
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
        source_path='',
        target_path='',
        type="allen",
        val_type="allen"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == "__main__":
    main()
