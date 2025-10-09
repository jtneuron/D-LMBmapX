import os

import SimpleITK as sitk


def read_3d_data(path):
    arr = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(arr)
    arr = arr[None, ...]
    return arr


def read_3d_data_normalize(path, max_value):
    arr = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(arr)
    if arr.max() > 1:
        arr = (arr - arr.min()) / (arr.max() - arr.min()) if (arr.max() - arr.min()) != 0 else arr
    arr = arr * max_value
    arr = arr[None, ...]
    return arr


def read_3d_mask(path, suffix=None):
    mask = []
    path_split = path.split(".")
    if suffix is not None:
        mask_path = path_split[0] + f"_{suffix}." + ".".join(path_split[1:])
        return read_3d_data(mask_path)

    mask_path = path_split[0] + "_label." + ".".join(path_split[1:])
    if os.path.exists(mask_path):
        mask = read_3d_data(mask_path)
    mask_path = path_split[0] + "_mask." + ".".join(path_split[1:])
    if mask == [] and os.path.exists(mask_path):
        mask = read_3d_data(mask_path)
    return mask

def read_edge(path, postfix):
    edge = []
    path_split = path.split(".")
    edge_path = path_split[0] + f"_{postfix}." + ".".join(path_split[1:])
    if os.path.exists(edge_path):
        edge = read_3d_data(edge_path)
    else:
        raise FileNotFoundError(f"Edge file {edge_path} not found")
    return edge
