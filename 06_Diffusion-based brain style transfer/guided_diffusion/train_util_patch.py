import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger, metrics
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import time
from .ModelSaver import ModelSaver
from skimage.metrics import peak_signal_noise_ratio
import pytorch_msssim


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            max_save_num=3,
            val_data=None,
            val_interval=1e4,
            continous=False,
            patch_size=64,
            stride=16,
            resolution=(512, 320),
    ):
        self.val_data = val_data
        self.val_interval = val_interval
        self.continous = continous
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size

        self.patch_size = patch_size
        self.stride = stride
        self.resolution = resolution

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self.model_saver = ModelSaver(max_save_num=max_save_num)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],  # dist_util.dev(), [int(os.environ["LOCAL_RANKRANK"])],
                output_device=dist_util.dev(),  # dist_util.dev()
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        start_time = time.time()
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond, _ = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            if self.step % self.log_interval == 0:
                consume_time = f"step {self.step} consume time: {time.time() - start_time}s"
                logger.log(consume_time)

            if self.val_data is not None and self.step % self.val_interval == 0:
                self.run_val()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if th.distributed.get_rank() == 0:
                self.save()

    def reassemble_patches(self, patches, patch_size, stride, original_size):
        h, w = original_size[1], original_size[0]  # 320,448
        h_patches = h // stride + (h % stride != 0)
        w_patches = w // stride + (w % stride != 0)

        # 初始化一个空白图像
        full_image = th.zeros((1, h, w))  #  torch.Size([1, 320, 448])

        # 将patches放回原位
        for i in range(h_patches):
            for j in range(w_patches):
                if i == h_patches - 1:
                    h_left, h_right = h - patch_size[1] - 1, -1
                else:
                    h_left, h_right = i * stride, i * stride + patch_size[1]
                if j == w_patches - 1:
                    w_left, w_right = w - patch_size[0] - 1, -1
                else:
                    w_left, w_right = j * stride, j * stride + patch_size[0]
                patch = th.unsqueeze(patches[i * w_patches + j], 0)
                full_image[:, h_left:h_right, w_left:w_right] = patch

        return full_image

    def \
            run_val(self):
        logger.dumpkvs()
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val", str(self.step))
        os.makedirs(result_path, exist_ok=True)
        for batch, cond, path in self.val_data:
            batch = batch.to(dist_util.dev())
            print("================path:",path)
            cond = {
                k: v.to(dist_util.dev())
                for k, v in cond.items()
            }
            idx += 1
            batch_size, in_channels, image_size = batch.shape[0], batch.shape[1], batch.shape[2:]
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(
                self.model,
                (batch_size, in_channels, image_size[0], image_size[1]),
                clip_denoised=True,
                model_kwargs=cond,
                progress=True,
                continous=self.continous
            )
            reconstructed_samples = []
            reconstructed_images = []
            reconstructed_conds = []
            for i in range(batch_size):
                '''
                image_patches.shape: torch.Size([6, 160, 224])
                sample_patches.shape: torch.Size([6, 160, 224])
                '''
                reconstructed_image = self.reassemble_patches(batch[i], patch_size=self.patch_size,
                                                              stride=self.stride, original_size=self.resolution)
                reconstructed_sample = self.reassemble_patches(sample[i], patch_size=self.patch_size,
                                                               stride=self.stride, original_size=self.resolution)
                reconstructed_cond = self.reassemble_patches(cond["cond_image"][i], patch_size=self.patch_size,
                                                             stride=self.stride, original_size=self.resolution)
                reconstructed_images.append(reconstructed_image)
                reconstructed_samples.append(reconstructed_sample)
                reconstructed_conds.append(reconstructed_cond)
            batch = th.stack(reconstructed_images)
            sample = th.stack(reconstructed_samples)
            cond["cond_image"] = th.stack(reconstructed_conds)

            if self.continous:
                continous_img = sample[1]
                sample = sample[0]
                continous_img = metrics.tensor2img(continous_img, nrow=batch_size)
                metrics.save_img(continous_img, '{}/{}_{}_continous_img.png'.format(result_path, self.step, idx))

            if cond.get("target_image", None) is not None:
                # concat_imgs = th.cat((cond["cond_image"], batch, cond["target_image"], sample), dim=-1)
                concat_imgs = th.cat((cond["cond_image"], batch, sample), dim=-1)
                targets = metrics.tensor2img(cond["target_image"])
            else:
                concat_imgs = th.cat((cond["cond_image"], batch, sample), dim=-1)
            # concat_imgs = th.cat((cond["cond_image"], batch, sample), dim=-1)

            concat_imgs = metrics.tensor2img(concat_imgs)
            cond_images = metrics.tensor2img(cond["cond_image"])
            sources = metrics.tensor2img(batch)
            translated = metrics.tensor2img(sample)
            metrics.save_img(concat_imgs, '{}/{}_{}_concatenate.png'.format(result_path, self.step, idx))
            metrics.save_img(cond_images, '{}/{}_{}_cond_images.png'.format(result_path, self.step, idx))
            metrics.save_img(sources, '{}/{}_{}_sources.png'.format(result_path, self.step, idx))
            metrics.save_img(translated, '{}/{}_{}_translated.png'.format(result_path, self.step, idx))
            if cond.get("target_image", None) is not None:
                metrics.save_img(targets, '{}/{}_{}_targets.png'.format(result_path, self.step, idx))
                _target_image = ((cond["target_image"] + 1) * 127.5)
                _sample = ((sample + 1) * 127.5)
                avg_psnr += peak_signal_noise_ratio(_target_image.cpu().numpy(), _sample.cpu().numpy(), data_range=255)
                avg_ssim += pytorch_msssim.ssim(_target_image, _sample, data_range=255).item()
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.log(f"{self.step} avg_psnr:{avg_psnr:.4e}")
        logger.log(f"{self.step} avg_ssim:{avg_ssim:.4e}")
        logger.logkv("avg_psnr", avg_psnr)
        logger.logkv("avg_ssim", avg_ssim)
        logger.dumpkvs()
        logger.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        val_log_path = os.path.join(os.getenv("OPENAI_LOGDIR"), "val_log.txt")
        with open(val_log_path, 'a') as f:
            f.write(f"{self.step} avg_psnr:{avg_psnr:.4e}\n")
            f.write(f"{self.step} avg_ssim:{avg_ssim:.4e}\n")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                # self.diffusion.training_losses,
                self.diffusion.training_losses_segbranch,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                    self.model_saver.post_handle(bf.join(get_blob_logdir(), filename))

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
                self.model_saver.post_handle(bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"))

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
