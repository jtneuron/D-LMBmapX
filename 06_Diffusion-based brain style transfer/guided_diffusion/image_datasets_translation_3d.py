import os

from torch.utils.data import Dataset
import json

import SimpleITK as sitk
import numpy as np


def read_3d_data(path):
    arr = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(arr)
    arr = arr[None, ...]
    return arr


def read_3d_mask(path):
    mask = []
    path_split = path.split(".")
    mask_path = path_split[0] + "_mask." + ".".join(path_split[1:])
    if os.path.exists(mask_path):
        mask = read_3d_data(mask_path)
    mask_path = path_split[0] + "_label." + ".".join(path_split[1:])
    if mask == [] and os.path.exists(mask_path):
        mask = read_3d_data(mask_path)
    return mask


class ImageDataset3D(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        with open(data_dir, 'r') as f:
            dataset_config = json.load(f)
            self.all_files = dataset_config['data']

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        volume = read_3d_data(self.all_files[idx])
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        volume = volume * 255.
        volume = volume.astype(np.float32)
        label = read_3d_mask(self.all_files[idx])
        img_name = os.path.basename(self.all_files[idx]).split(".")[0]
        img = {"img_name": img_name, "volume": volume, "label": label}
        return img
