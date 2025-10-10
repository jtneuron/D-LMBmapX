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
import cv2

from guided_diffusion.data_util import get_cond_image, get_cond_image_fda


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
        in_channels=3,
        patch_size=(224, 160),
        stride=160,
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
    dataset = ImagePatchDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        in_channels=in_channels,
        patch_size=patch_size,
        stride=stride,
    )
    print(f"datasets size : {len(dataset)}")
    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            pin_memory=True,  # 直接加载到显存中，达到加速效果
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # False,  # former: True
            num_workers=1,
            drop_last=True,
            pin_memory=True,  # 直接加载到显存中，达到加速效果
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


class ImagePatchDataset(Dataset):
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
            patch_size=(224, 160),
            stride=160,
    ):
        super().__init__()
        if not isinstance(resolution, tuple):
            resolution = (resolution, resolution)
        self.resolution = resolution
        local_paths = [i[0] for i in image_paths]

        self.local_images = local_paths[shard:][::num_shards]
        self.target_images = None
        if len(image_paths[0]) > 1:
            target_paths = [i[1] for i in image_paths]
            self.target_images = target_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        local_image_path = self.local_images[idx]
        local_image = self.get_image_from_path(local_image_path)
        cond_image = get_cond_image(local_image)
        mask_image = self.get_mask_image(local_image_path)
        if self.target_images is not None:
            target_image_path = self.target_images[idx]
            target_image = self.get_image_from_path(target_image_path)

        # to_tensor
        local_image = torchvision.transforms.ToTensor()(local_image)  # torch.Size([1, 320, 448])
        cond_image = torchvision.transforms.ToTensor()(cond_image)
        if mask_image is not None:
            mask_image = torch.from_numpy(np.array(mask_image)).unsqueeze(dim=0)
        if self.target_images is not None:
            target_image = torchvision.transforms.ToTensor()(target_image)

        # patch_num
        h, w = local_image.shape[1], local_image.shape[2]
        h_patches = h // self.stride + (h % self.stride != 0)
        w_patches = w // self.stride + (w % self.stride != 0)
        # patch crop
        image_patches = []
        cond_patches = []
        mask_patches = []
        target_patches = []
        for i in range(h_patches):
            for j in range(w_patches):
                if i == h_patches - 1:
                    h_left, h_right = h - self.patch_size[1] - 1, -1
                else:
                    h_left, h_right = i * self.stride, i * self.stride + self.patch_size[1]
                if j == w_patches - 1:
                    w_left, w_right = w - self.patch_size[0] - 1, -1
                else:
                    w_left, w_right = j * self.stride, j * self.stride + self.patch_size[0]

                image_patch = local_image[:, h_left:h_right, w_left:w_right]  # torch.Size([1, 160, 224])
                image_patches.append(image_patch)
                cond_patch = cond_image[:, h_left:h_right, w_left:w_right]
                cond_patches.append(cond_patch)
                if mask_image is not None:
                    mask_patch = mask_image[:, h_left:h_right, w_left:w_right]
                    mask_patches.append(mask_patch)
                if self.target_images is not None:
                    target_patch = target_image[:, h_left:h_right, w_left:w_right]
                    target_patches.append(target_patch)
        image_patches = torch.squeeze(torch.stack(image_patches), 1)  # torch.Size([6, 160, 224])
        cond_patches = torch.squeeze(torch.stack(cond_patches), 1)
        imgs_list = [image_patches, cond_patches]
        if self.target_images is not None:
            target_patches = torch.squeeze(torch.stack(target_patches), 1)
            imgs_list.append(target_patches)
        if mask_image is not None:
            mask_patches = torch.squeeze(torch.stack(mask_patches), 1)
            imgs_list.append(mask_patches)

        imgs_list = self.transform_augment(imgs_list)

        if mask_image is not None:
            mask_image = imgs_list.pop()
        min_max = (-1, 1)
        imgs_list = [img * (min_max[1] - min_max[0]) + min_max[0]
                     for img in imgs_list]
        if self.target_images is not None:
            image_patches, cond_patches, target_patches = imgs_list
        else:
            image_patches, cond_patches = imgs_list  # image_patches.shape: torch.Size([6, 1, 160, 224])

        out_dict = {"cond_image": cond_patches, "original_image": image_patches}
        if mask_image is not None:
            out_dict["mask_image"] = mask_patches
        if self.target_images is not None:
            out_dict["target_image"] = target_patches
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return image_patches, out_dict, local_image_path

    def get_image_from_path(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        if self.in_channels == 3:
            pil_image = pil_image.convert("RGB")
        else:
            pil_image = pil_image.convert("L")
        # if pil_image.size != self.resolution:
        #     pil_image = pil_image.resize(
        #         self.resolution, Image.BICUBIC)
        return pil_image

    def transform_augment(self, imgs):
        if self.random_flip is True:
            imgs = torch.stack(imgs, 0)
            imgs = torchvision.transforms.RandomHorizontalFlip()(imgs)
            imgs = torch.unbind(imgs, dim=0)
        return list(imgs)

    def get_mask_image(self, path):
        dirname, basename = os.path.split(path)

        dirname = dirname.replace('data_', 'mask_all_')
        mask_path = os.path.join(dirname, basename)

        if not os.path.exists(mask_path):
            return None

        with bf.BlobFile(mask_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("L")
        # if pil_image.size != self.resolution:
        #     pil_image = pil_image.resize(
        #         self.resolution, Image.NEAREST)
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
        patch_size=(224, 160),
        stride=160,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    with open(data_dir, 'r') as f:
        dataset_config = json.load(f)
        all_files = dataset_config['data']
    all_files = all_files[:data_num]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImagePatchDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        in_channels=in_channels,
        patch_size=patch_size,
        stride=stride,
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
