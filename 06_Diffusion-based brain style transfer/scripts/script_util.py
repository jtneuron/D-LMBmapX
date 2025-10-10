import torch
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from guided_diffusion.data_util import get_cond_image
from scripts.metric_util import dice_metric


def get_cond(imgs, min_max=(-1, 1), isTrain=False, typ="x"):
    imgs = (imgs + 1) * 127.5
    res_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img.squeeze(axis=0)
        cond_image = get_cond_image(img, isTrain=isTrain, type=typ)
        cond_image = torchvision.transforms.ToTensor()(cond_image)
        cond_image = cond_image * (min_max[1] - min_max[0]) + min_max[0]
        res_imgs.append(cond_image)
    cond_images = torch.stack(res_imgs, dim=0)
    # cond_images = torch.cat(res_imgs, dim=0)
    # cond_images = cond_images.unsqueeze(dim=1)
    return {"cond_image": cond_images}


def get_dice_and_mask(labels, predict, num_classes):
    predict_softmax = F.softmax(predict, dim=1)
    predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long(),
                                num_classes=num_classes).permute((0, 3, 1, 2)).contiguous()
    label_one_hot = F.one_hot(labels.squeeze(dim=1).long(), num_classes=num_classes).permute(
        (0, 3, 1, 2)).contiguous()
    _dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach(), reduction="none")
    predict_mask = torch.argmax(predict_softmax, dim=1).unsqueeze(dim=1)
    # normalize to [-1,1]
    predict_mask = (predict_mask / num_classes) * 2. - 1.
    return _dice, predict_mask


def post_process(sample, source):
    # (-1,1) to (0,2)
    sample = sample + 1.
    foreground = torch.where(source > -1., 1., 0.)
    sample = sample * foreground
    # (0,2) to (-1,1)
    sample = sample - 1.
    return sample


def set_random_seed(seed=0):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def save_img(img, output_path, input_img_path, identifier):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = (img + 1.) / 2.
    img = img * 255.
    img = img.astype(np.uint8)
    volume_name = os.path.basename(os.path.dirname(input_img_path))
    basename = os.path.basename(input_img_path)
    path = os.path.join(output_path, identifier, volume_name, basename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(img)
    img.save(path)
