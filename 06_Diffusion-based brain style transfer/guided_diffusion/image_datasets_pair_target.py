import math
import random

import torchvision
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import torch
import os


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

        source_paths = [i[0] for i in image_paths]
        cond_paths = [i[1] for i in image_paths]
        self.local_images = source_paths[shard:][::num_shards]
        self.cond_images = cond_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        local_image_path = self.local_images[idx]
        cond_image_path = self.cond_images[idx]

        local_image = self.get_image_from_path(local_image_path)
        cond_image = self.get_image_from_path(cond_image_path)

        mask_image = self.get_mask_image(local_image_path)

        imgs_list = [local_image, cond_image]
        imgs_list = [torchvision.transforms.ToTensor()(img) for img in imgs_list]

        if mask_image is not None:
            mask_image = np.array(mask_image)
            mask_image = torch.from_numpy(mask_image)
            mask_image = mask_image.unsqueeze(dim=0)
            imgs_list.append(mask_image)

        imgs_list = self.transform_augment(imgs_list)

        if mask_image is not None:
            mask_image = imgs_list.pop()

        min_max = (-1, 1)
        imgs_list = [img * (min_max[1] - min_max[0]) + min_max[0]
                     for img in imgs_list]

        local_image, cond_image = imgs_list

        out_dict = {"cond_image": cond_image, "target_image": local_image}
        
        # out_dict['image_path'] = local_image_path
        
        if mask_image is not None:
            out_dict["mask_image"] = mask_image

        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return local_image, out_dict

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
    all_files = all_files[:data_num]
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
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader
