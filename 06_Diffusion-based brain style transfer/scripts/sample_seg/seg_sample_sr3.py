"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import sys

sys.path.append(".")

import torch as th
from guided_diffusion.image_datasets_pair import load_val_data
from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from skimage.metrics import peak_signal_noise_ratio
import pytorch_msssim
import numpy as np
from scripts.script_util import get_dice_and_mask, post_process
import torch


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
    logger.log(f"seg_model_path: {args.seg_model_path}")
    seg_model = torch.load(args.seg_model_path)
    seg_model.to(dist_util.dev())
    seg_model.eval()
    num_classes = seg_model.out_channels
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

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    val_dice = []
    count = [0] * (num_classes - 1)
    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val")
    os.makedirs(result_path, exist_ok=True)
    for batch, cond in data:
        batch = batch.to(dist_util.dev())
        cond = {
            k: v.to(dist_util.dev())
            for k, v in cond.items()
        }
        idx += 1
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

        if args.continous:
            continous_img = sample[1]
            sample = sample[0]
            continous_img = metrics.tensor2img(continous_img, nrow=batch_size)
            metrics.save_img(continous_img, '{}/{}_continous_img.png'.format(result_path, idx))
        if cond.get("target_image", None) is not None:
            concat_imgs = th.cat((batch, cond["target_image"], sample), dim=-1)
            targets = metrics.tensor2img(cond["target_image"])
        else:
            concat_imgs = th.cat((batch, sample), dim=-1)
        # concat_imgs = metrics.tensor2img(concat_imgs, nrow=2)
        cond_images = metrics.tensor2img(cond["cond_image"])
        sources = metrics.tensor2img(batch)
        translated = metrics.tensor2img(sample)

        metrics.save_img(cond_images, '{}/{}_cond_images.png'.format(result_path, idx))
        metrics.save_img(sources, '{}/{}_sources.png'.format(result_path, idx))
        metrics.save_img(translated, '{}/{}_translated.png'.format(result_path, idx))

        if cond.get("target_image", None) is not None:
            metrics.save_img(targets, '{}/{}_targets.png'.format(result_path, idx))
            _target_image = ((cond["target_image"] + 1) * 127.5)
            _sample = ((sample + 1) * 127.5)
            avg_psnr += peak_signal_noise_ratio(_target_image.cpu().numpy(), _sample.cpu().numpy(), data_range=255)
            avg_ssim += pytorch_msssim.ssim(_target_image, _sample, data_range=255)

        with torch.no_grad():
            labels = cond["mask_image"]
            t = th.zeros(sample.shape[0], dtype=th.long, device=dist_util.dev())
            predict = seg_model(sample, timesteps=t)
            _dice, predict_mask = get_dice_and_mask(labels, predict, num_classes)
            val_dice.append(_dice.cpu().numpy().tolist())

            _labels = (labels / num_classes) * 2. - 1.
            concat_imgs = th.cat((concat_imgs, _labels, predict_mask), dim=-1)
            concat_imgs = metrics.tensor2img(concat_imgs, nrow=1)
            metrics.save_img(concat_imgs, '{}/{}_concat.png'.format(result_path, idx))
            for mask_id in torch.unique(labels):
                if mask_id:
                    count[mask_id - 1] += 1
    count = [1 if num == 0 else num for num in count]
    count = np.array(count)
    val_dice = np.array(val_dice)
    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    index_region = {1: 'hpf', 2: 'cp', 3: 'bs', 4: 'cbx', 5: 'ctx'}

    val_region = np.sum(val_dice, axis=0) / count
    for i in range(val_region.shape[0]):
        logger.logkv(f"{index_region[i + 1]}_dice", val_region[i])

    avg_dice = np.mean(val_region).item()

    logger.log(count)
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    logger.logkv("avg_psnr", avg_psnr)
    logger.logkv("avg_ssim", avg_ssim)
    logger.logkv("avg_dice", avg_dice)
    logger.dumpkvs()
    logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val_log_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val_log.txt")
    with open(val_log_path, 'a') as f:
        f.write(f" avg_psnr:{avg_psnr:.4e}\n")
        f.write(f" avg_ssim:{avg_ssim:.4e}\n")
        f.write(f" avg_dice:{avg_dice:.4e}\n")
        for i in range(val_region.shape[0]):
            f.write(f"{index_region[i + 1]}_dice: {val_region[i]:.4e}\n")
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
        seg_model_path='',

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == "__main__":
    main()
