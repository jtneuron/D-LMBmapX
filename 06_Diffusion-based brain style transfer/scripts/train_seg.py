"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys

sys.path.append(".")
import blobfile as bf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam

from guided_diffusion import dist_util, logger, metrics
from guided_diffusion.ModelSaver import ModelSaver
from guided_diffusion.image_datasets_seg import load_data, load_val_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_seg_and_diffusion, seg_and_diffusion_defaults,
)
from guided_diffusion.train_util import parse_resume_step_from_filename

from scripts.script_util import set_random_seed
import numpy as np
import warnings

from scripts.loss_util import DiceLoss
from scripts.metric_util import dice_metric

warnings.filterwarnings("ignore")


def eval(args):
    os.environ['OPENAI_LOGDIR'] = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    os.makedirs(args.logdir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure()

    model, diffusion = create_seg_and_diffusion(
        **args_to_dict(args, seg_and_diffusion_defaults().keys())
    )
    logger.log(list(vars(args).items()))
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )
    model = torch.load(args.resume_checkpoint)
    model.to(dist_util.dev())
    dist_util.sync_params(model.parameters())
    val_data = load_val_data(
        data_dir=args.val_data_dir,
        batch_size=args.val_batch_size,
        image_size=args.resolution,
        class_cond=False,
        in_channels=args.in_channels,
        data_num=args.val_data_num,
    )
    num_classes = model.out_channels
    model.eval()
    idx = 0
    val_dice = [[] for _ in range(num_classes - 1)]
    result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "imgs")
    os.makedirs(result_path, exist_ok=True)
    with torch.no_grad():
        for batch, extra in val_data:
            batch = batch.to(dist_util.dev())
            labels = extra["mask_image"].to(dist_util.dev())
            # Noisy images
            if args.noised:
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
                batch = diffusion.q_sample(batch, t)
            else:
                t = torch.zeros(batch.shape[0], dtype=torch.long, device=dist_util.dev())

            predict = model(batch, timesteps=t)

            predict_softmax = F.softmax(predict, dim=1)
            label_one_hot = F.one_hot(labels.squeeze(dim=1).long(), num_classes=num_classes).permute(
                (0, 3, 1, 2)).contiguous()
            predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long(),
                                        num_classes=num_classes).permute((0, 3, 1, 2)).contiguous()

            _dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach(), reduction="none")
            for mask_id in torch.unique(labels).cpu().numpy().tolist():
                if mask_id:
                    val_dice[mask_id-1].append(_dice[mask_id - 1].cpu().item())

            _predict = torch.argmax(predict_softmax, dim=1).unsqueeze(dim=1)
            _predict = (_predict / num_classes) * 2. - 1.
            _labels = (labels / num_classes) * 2. - 1.
            concat_imgs = torch.cat((batch, _labels, _predict), dim=-1)
            concat_imgs = metrics.tensor2img(concat_imgs, nrow=2)
            metrics.save_img(concat_imgs, '{}/{}_concat.png'.format(result_path, idx))
            idx += 1

    # {'bs': 8, 'cbx': 9, 'cp': 6, 'ctx': 10, 'hpf': 5}
    # label_map = {0: 0, 5: 1, 6: 2, 8: 3, 9: 4, 10: 5
    index_region = {1: 'hpf', 2: 'cp', 3: 'bs', 4: 'cbx', 5: 'ctx'}

    mean_val_region, std_val_region = [], []

    for i in range(len(val_dice)):
        mean_val_region.append(np.mean(val_dice[i]) if len(val_dice[i]) > 0 else 0.)
        std_val_region.append(np.std(val_dice[i]) if len(val_dice[i]) > 0 else 0.)

    mean_val_region = np.array(mean_val_region)
    std_val_region = np.array(std_val_region)

    mean_val_dice = np.mean(mean_val_region)
    logger.logkv("avg_dice", mean_val_dice)

    for i in range(mean_val_region.shape[0]):
        logger.logkv(f"{index_region[i + 1]}_dice", mean_val_region[i])

    logger.dumpkvs()
    for i in range(std_val_region.shape[0]):
        logger.logkv(f"{index_region[i + 1]}_dice_std", std_val_region[i])

    logger.dumpkvs()
    dist.barrier()


def main():
    set_random_seed(0)
    args = create_argparser().parse_args()
    args.resolution = tuple(args.resolution)
    if args.eval:
        eval(args)
        return
    os.environ['OPENAI_LOGDIR'] = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    os.makedirs(args.logdir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure()
    model, diffusion = create_seg_and_diffusion(
        **args_to_dict(args, seg_and_diffusion_defaults().keys())
    )
    logger.log(list(vars(args).items()))
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model = torch.load(args.resume_checkpoint)
            model.to(dist_util.dev())
            # model.load_state_dict(
            #     dist_util.load_state_dict(
            #         args.resume_checkpoint, map_location=dist_util.dev()
            #     )
            # )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.resolution,
        class_cond=False,
        random_crop=True,
        in_channels=args.in_channels,
    )
    if args.val_data_dir:
        val_data = load_val_data(
            data_dir=args.val_data_dir,
            batch_size=args.val_batch_size,
            image_size=args.resolution,
            class_cond=False,
            in_channels=args.in_channels,
            data_num=args.val_data_num,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(
            f"loading optimizer state from checkpoint: {opt_checkpoint}")
        optimizer.load_state_dict(
            dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training seg model...")
    num_classes = args.out_channels
    loss_function = DiceLoss(background=args.background, reduction='mean')
    model_saver = ModelSaver(max_save_num=args.max_save_num)
    step = 0
    # writer = SummaryWriter(log_dir=os.path.join(os.environ['OPENAI_LOGDIR'], "tb"))
    best_val_dice = 0.
    for epoch in range(1, args.epoches + 1):
        model.train()
        train_losses = []
        train_dice = []

        if args.anneal_lr and epoch > 100:
            set_annealed_lr(
                optimizer, args.lr, epoch / args.epoches)

        for batch, extra in data:
            batch = batch.to(dist_util.dev())
            labels = extra["mask_image"].to(dist_util.dev())
            optimizer.zero_grad()
            # Noisy images
            if args.noised:
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
                batch = diffusion.q_sample(batch, t)
            else:
                t = torch.zeros(batch.shape[0], dtype=torch.long, device=dist_util.dev())
            predict = model(batch, timesteps=t)
            predict_softmax = F.softmax(predict, dim=1)
            label_one_hot = F.one_hot(labels.squeeze(dim=1).long(), num_classes=num_classes).permute(
                (0, 3, 1, 2)).contiguous()
            loss = loss_function(predict_softmax, label_one_hot.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long(),
                                        num_classes=num_classes).permute((0, 3, 1, 2)).contiguous()

            _dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach())
            train_dice.append(_dice.item())
            step += 1
        mean_train_loss = np.mean(train_losses)
        mean_train_region = np.sum(train_dice, axis=0) / np.count_nonzero(train_dice, axis=0)
        mean_train_dice = np.mean(mean_train_region)
        logger.logkv("epoch", epoch)
        logger.logkv("train_loss", mean_train_loss)
        logger.logkv("train_dice", mean_train_dice)

        # print("-------------------------------------")
        # print(f'training loss: {mean_train_loss}')
        # print(f"training dice:  {mean_train_dice}")
        # writer.add_scalar("train/loss", mean_train_loss, epoch)
        # writer.add_scalar("train/dice", mean_train_dice, epoch)

        if val_data is not None and not epoch % args.val_interval:
            model.eval()
            val_losses = []
            val_dice = []
            with torch.no_grad():
                for batch, extra in val_data:
                    batch = batch.to(dist_util.dev())
                    labels = extra["mask_image"].to(dist_util.dev())
                    # Noisy images
                    if args.noised:
                        t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
                        batch = diffusion.q_sample(batch, t)
                    else:
                        t = torch.zeros(batch.shape[0], dtype=torch.long, device=dist_util.dev())

                    predict = model(batch, timesteps=t)

                    predict_softmax = F.softmax(predict, dim=1)
                    label_one_hot = F.one_hot(labels.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        (0, 3, 1, 2)).contiguous()
                    loss = loss_function(predict_softmax, label_one_hot.float())
                    val_losses.append(loss.item())
                    predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long(),
                                                num_classes=num_classes).permute((0, 3, 1, 2)).contiguous()

                    _dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach())
                    val_dice.append(_dice.item())
            mean_val_loss = np.mean(val_losses)

            mean_val_region = np.sum(val_dice, axis=0) / np.count_nonzero(val_dice, axis=0)
            mean_val_dice = np.mean(mean_val_region)

            logger.logkv("val_loss", mean_val_loss)
            logger.logkv("val_dice", mean_val_dice)
            if mean_val_dice.item() > best_val_dice:
                best_val_dice = mean_val_dice.item()
                torch.save(
                    model,
                    os.path.join(logger.get_dir(), f"best_epoch.pt"),
                )

            # print("-------------------------------------")
            # print(f'val loss: {mean_val_loss}')
            # print(f"val dice:  {mean_val_dice}")
            # writer.add_scalar("val/loss", mean_val_loss, epoch)
            # writer.add_scalar("val/dice", mean_val_dice, epoch)
        logger.dumpkvs()
        if (dist.get_rank() == 0
                and not epoch % args.save_interval
        ):
            logger.log("saving model...")
            save_model(model, optimizer, epoch, model_saver)

    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(model, opt, step, model_saver):
    if dist.get_rank() == 0:
        torch.save(
            model,
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        # )

        torch.save(opt.state_dict(), os.path.join(
            logger.get_dir(), f"opt{step:06d}.pt"))
        model_saver.post_handle(os.path.join(logger.get_dir(), f"model{step:06d}.pt"))
        model_saver.post_handle(os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def create_argparser():
    defaults = dict(
        data_dir="",
        out_channels=2,
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        logdir="./output/temp",
        gpu='0',
        max_save_num=3,
        val_batch_size=1,
        val_data_num=16,
        val_interval=10000,
        epoches=100,
        background=True,
        eval=False,
    )
    defaults.update(seg_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(f"--resolution", nargs='+', type=int)
    return parser


if __name__ == '__main__':
    main()
