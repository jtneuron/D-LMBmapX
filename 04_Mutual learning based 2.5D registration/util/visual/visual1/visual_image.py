import math

# import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def cat_imgs(img_list, interval, view_dim):
    shape = img_list[0].shape
    if view_dim == 2:
        img_slice = torch.cat(
            [torch.cat([img[:, :, i, :, :] for img in img_list], dim=-1) for i in range(0, shape[2], interval)], dim=0)
    elif view_dim == 3:
        img_slice = torch.cat(
            [torch.cat([img[:, :, :, i, :] for img in img_list], dim=-1) for i in range(0, shape[3], interval)], dim=0)
    elif view_dim == 4:
        img_slice = torch.cat(
            [torch.cat([img[:, :, :, :, i] for img in img_list], dim=-1) for i in range(0, shape[4], interval)], dim=0)
    else:
        raise Exception()
    return img_slice


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1), nrow=-1):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if nrow == -1:
        n_img = len(tensor)
        nrow = int(math.sqrt(n_img))

    if n_dim == 4:
        img_np = make_grid(tensor, nrow=nrow, normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


# def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)
