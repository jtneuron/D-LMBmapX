import math
import os
import random

import torchvision
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import torch

from guided_diffusion.data_util import get_cond_image
from . import metrics
from einops import rearrange


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
        in_channels=3
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    with open(data_dir, 'r') as f:
        dataset_config = json.load(f)
        all_files = dataset_config['data']
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        in_channels=in_channels
    )
    print(f"datasets size : {len(dataset)}")
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        name = entry.split(".")[0]
        if name.endswith("_mask"):
            continue
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            in_channels=3,
    ):
        super().__init__()
        if not isinstance(resolution, tuple):
            resolution = (resolution, resolution)
        self.resolution = resolution

        local_paths = []  # current image
        for i in range(len(image_paths)):
            num = int(os.path.splitext(os.path.basename(image_paths[i][0]))[0])
            if num != 0 and num != (512 - 1):
                local_paths.append(image_paths[i][0])
            elif num == 0:
                local_paths.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num + 1) + os.path.splitext(image_paths[i][0])[-1]))
            elif num == (512 - 1):
                local_paths.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num - 1) + os.path.splitext(image_paths[i][0])[-1]))

        local_paths_previous = []  # previous image
        for i in range(len(image_paths)):
            num = int(os.path.splitext(os.path.basename(image_paths[i][0]))[0])
            if num != 0 and num != (512 - 1):
                local_paths_previous.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num - 1) + os.path.splitext(image_paths[i][0])[-1]))
            elif num == 0:
                local_paths_previous.append(image_paths[i][0])
            elif num == (512 - 1):
                local_paths_previous.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num - 2) + os.path.splitext(image_paths[i][0])[-1]))

        local_paths_after = []  # after image
        for i in range(len(image_paths)):
            num = int(os.path.splitext(os.path.basename(image_paths[i][0]))[0])
            if num != (512 - 1) and num != 0:
                local_paths_after.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num + 1) + os.path.splitext(image_paths[i][0])[-1]))
            elif num == (512 - 1):
                local_paths_after.append(image_paths[i][0])
            elif num == 0:
                local_paths_after.append(os.path.join(os.path.dirname(image_paths[i][0]),
                                                "%04d" % (num + 2) + os.path.splitext(image_paths[i][0])[-1]))

        self.local_images = local_paths[shard:][::num_shards]
        self.local_images_previous = local_paths_previous[shard:][::num_shards]
        self.local_images_after = local_paths_after[shard:][::num_shards]

        self.target_images = None
        if len(image_paths[0]) > 1:
            target_paths = []  # current
            for i in range(len(image_paths)):
                num = int(os.path.splitext(os.path.basename(image_paths[i][1]))[0])
                if num != 0 and num != (512 - 1):
                    target_paths.append(image_paths[i][1])
                elif num == 0:
                    target_paths.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num + 1) + os.path.splitext(image_paths[i][1])[-1]))
                elif num == (512 - 1):
                    target_paths.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num - 1) + os.path.splitext(image_paths[i][1])[-1]))
            target_paths_previous = []  # previous image
            for i in range(len(image_paths)):
                num = int(os.path.splitext(os.path.basename(image_paths[i][1]))[0])
                if num != 0 and num != (512 - 1):
                    target_paths_previous.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num - 1) + os.path.splitext(image_paths[i][1])[-1]))
                elif num == 0:
                    target_paths_previous.append(image_paths[i][1])
                elif num == (512 - 1):
                    target_paths_previous.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num - 2) + os.path.splitext(image_paths[i][1])[-1]))
            target_paths_after = []  # after image
            for i in range(len(image_paths)):
                num = int(os.path.splitext(os.path.basename(image_paths[i][1]))[0])
                if num != (512 - 1) and num != 0:
                    target_paths_after.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num + 1) + os.path.splitext(image_paths[i][1])[-1]))
                elif num == (512 - 1):
                    target_paths_after.append(image_paths[i][1])
                elif num == 0:
                    target_paths_after.append(os.path.join(os.path.dirname(image_paths[i][1]),
                                                "%04d" % (num + 2) + os.path.splitext(image_paths[i][1])[-1]))
            self.target_images = target_paths[shard:][::num_shards]
            self.target_images_previous = target_paths_previous[shard:][::num_shards]
            self.target_images_after = target_paths_after[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # current
        local_image_path = self.local_images[idx]
        local_image = self.get_image_from_path(local_image_path)
        cond_image = get_cond_image(local_image)
        mask_image = self.get_mask_image(local_image_path)
        # previous
        local_image_path_previous = self.local_images_previous[idx]
        local_image_previous = self.get_image_from_path(local_image_path_previous)
        cond_image_previous = get_cond_image(local_image_previous)
        mask_image_previous = self.get_mask_image(local_image_path_previous)
        # after
        local_image_path_after = self.local_images_after[idx]
        local_image_after = self.get_image_from_path(local_image_path_after)
        cond_image_after = get_cond_image(local_image_after)
        mask_image_after = self.get_mask_image(local_image_path_after)

        imgs_list = [local_image, local_image_previous, local_image_after, cond_image, cond_image_previous,
                     cond_image_after]

        if self.target_images is not None:
            target_image_path = self.target_images[idx]
            target_image = self.get_image_from_path(target_image_path)
            target_image_path_previous = self.target_images_previous[idx]
            target_image_previous = self.get_image_from_path(target_image_path_previous)
            target_image_path_after = self.target_images_after[idx]
            target_image_after = self.get_image_from_path(target_image_path_after)
            imgs_list.append(target_image, target_image_previous, target_image_after)

        imgs_list = [torchvision.transforms.ToTensor()(img) for img in imgs_list]

        if mask_image is not None:
            mask_image = np.array(mask_image)
            mask_image = torch.from_numpy(mask_image)
            mask_image = mask_image.unsqueeze(dim=0)
            mask_image_previous = np.array(mask_image_previous)
            mask_image_previous = torch.from_numpy(mask_image_previous)
            mask_image_previous = mask_image_previous.unsqueeze(dim=0)
            mask_image_after = np.array(mask_image_after)
            mask_image_after = torch.from_numpy(mask_image_after)
            mask_image_after = mask_image_after.unsqueeze(dim=0)
            imgs_list.append(mask_image, mask_image_previous, mask_image_after)

        imgs_list = self.transform_augment(imgs_list)

        if mask_image is not None:
            mask_image_after = imgs_list.pop()
            mask_image_previous = imgs_list.pop()
            mask_image = imgs_list.pop()

        min_max = (-1, 1)
        imgs_list = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs_list]

        if self.target_images is not None:
            local_image, local_image_previous, local_image_after, cond_image, cond_image_previous, cond_image_after, target_image, target_image_previous, target_image_after = imgs_list
        else:
            local_image, local_image_previous, local_image_after, cond_image, cond_image_previous, cond_image_after = imgs_list

        image = torch.cat((local_image_previous, local_image, local_image_after), dim=0)
        cond = torch.cat((cond_image_previous, cond_image, cond_image_after), dim=0)

        out_dict = {"cond_image": cond, "original_image": image}
        # out_dict['image_path'] = local_image_path

        if self.target_images is not None:
            target = torch.cat((target_image_previous, target_image, target_image_after), dim=0)
            out_dict["target_image"] = target

        if mask_image is not None:
            mask = torch.cat((mask_image_previous, mask_image, mask_image_after), dim=0)
            out_dict["mask_image"] = mask

        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # cond_imgs = rearrange(cond, '(c t) x y -> c (t x) y', t=3)
        # source_imgs = rearrange(image, '(c t) x y -> c (t x) y', t=3)
        # cond_imgs = metrics.tensor2img(cond_imgs)
        # source_imgs = metrics.tensor2img(source_imgs)
        # metrics.save_img(cond_imgs, '{}/{}_cond_img.png'.format('/media/user/hdd1/liuhe/i2i_net/FGDM/output/temp/', idx))
        # metrics.save_img(source_imgs, '{}/{}_source_img.png'.format('/media/user/hdd1/liuhe/i2i_net/FGDM/output/temp/', idx))
        # print(len(source_imgs[source_imgs != 0]), len(source_imgs[source_imgs == 255]))

        image_paths =  [local_image_path_previous, local_image_path, local_image_path_after]

        return image, out_dict, image_paths

    def get_image_from_path(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        if self.in_channels == 3:
            pil_image = pil_image.convert("RGB")
        else:
            pil_image = pil_image.convert("L")
        if pil_image.size != self.resolution:
            pil_image = pil_image.resize(
                self.resolution, Image.BICUBIC)
        return pil_image

    def transform_augment(self, imgs):
        if self.random_flip is True:
            imgs = torch.stack(imgs, 0)
            imgs = torchvision.transforms.RandomHorizontalFlip()(imgs)
            imgs = torch.unbind(imgs, dim=0)
        return list(imgs)

    def get_mask_image(self, path):
        basename = os.path.basename(path)
        name_split = basename.split(".")
        mask_name = name_split[0] + "_mask." + ".".join(name_split[1:])
        mask_path = os.path.join(os.path.dirname(path), mask_name)

        if not os.path.exists(mask_path):
            return None

        with bf.BlobFile(mask_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("L")
        if pil_image.size != self.resolution:
            pil_image = pil_image.resize(
                self.resolution, Image.NEAREST)
        return pil_image


def load_val_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        in_channels=3,
        data_num=-1,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    with open(data_dir, 'r') as f:
        dataset_config = json.load(f)
        all_files = dataset_config['data']
    # all_files = all_files[:data_num]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        in_channels=in_channels
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    return loader


def load_source_data_for_domain_translation(
        *,
        batch_size,
        image_size,
        data_dir="./experiments/imagenet",
        in_channels=3,
        class_cond=True
):
    """
    This function is new in DDIBs: loads the source dataset for translation.
    For the dataset, create a generator over (images, kwargs) pairs.
    No image cropping, flipping or shuffling.

    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = [f for f in _list_image_files_recursively(
        data_dir) if "translated" not in f]
    # Classes are the first part of the filename, before an underscore: e.g. "291_1.png"
    classes = None
    if class_cond:
        classes = [int(bf.basename(path).split("_")[0]) for path in all_files]
    dataset = ImageDataset(
        image_size,
        all_files,
        in_channels=in_channels,
        random_flip=False,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=1)
    yield from loader


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(
        min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
