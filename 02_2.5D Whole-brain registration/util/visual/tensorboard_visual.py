import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from util.visual.visual_image import preview_image
from util.visual.visual_registration import preview_3D_deformation, preview_3D_vector_field, RGB_dvf, PlotGrid_3d, \
    comput_fig


def tensorboard_visual_segmentation(mode, name, writer, step, volume, predict, target, interval=10):
    """
    mode: train/val/test
    name: image name
    writer: SummaryWriter
    predict:  D, H, W
    target:   D, H, W
    """

    title_name = name + '_volume, label, predict'

    img_list = [volume, target, predict]
    tag = mode + '/' + name
    visual_img_list(tag=tag, title_name=title_name, writer=writer, step=step, img_list=img_list, interval=interval)


def tensorboard_visual_registration(mode, name, writer, step, fix, mov, reg, interval=10):
    """
    mode: train/val/test
    name: image name
    writer: SummaryWriter
    predict:  D, H, W
    target:   D, H, W
    """

    title_name = name + '_fix, mov, reg'
    img_list = [fix, mov, reg]
    tag = mode + '/' + name
    visual_img_list(tag=tag, title_name=title_name, writer=writer, step=step, img_list=img_list, interval=interval)


def tensorboard_visual_mae(mode, name, writer, step, img, mask_img, pred_img, interval=10):
    """
    mode: train/val/test
    name: image name
    writer: SummaryWriter
    predict:  D, H, W
    target:   D, H, W
    """

    title_name = name + '_img, mask_img, pred_img'
    img_list = [img, mask_img, pred_img]
    tag = mode + '/' + name
    visual_img_list(tag=tag, title_name=title_name, writer=writer, step=step, img_list=img_list, interval=interval)


def visual_img_list(tag, title_name, writer, step, img_list, interval):
    # 1, H*rows_number, W*2
    shape = img_list[0].shape

    img_slice = torch.cat(
        [torch.cat([img[i, :, :].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[0], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/D', torch.cat((title_patch, img_slice), dim=1), step)

    # 1, D*rows_number, W*2
    img_slice = torch.cat(
        [torch.cat([img[:, i, :].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[1], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/H', torch.cat((title_patch, img_slice), dim=1), step)

    # 1, D*rows_number, H*2
    img_slice = torch.cat(
        [torch.cat([img[:, :, i].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[2], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/W', torch.cat((title_patch, img_slice), dim=1), step)


def create_header(shape, name):
    header = np.zeros((100, shape[2], 1), dtype=np.uint8) + 255
    header = cv2.putText(header, name, (10, header.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, 0, 1)
    header = header.astype(np.float32) / 255
    header = np.transpose(header, (2, 0, 1))
    header = torch.Tensor(header)
    return header


def visual_gradient(model: Module, writer: SummaryWriter, step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram("grad/" + name, param.grad, step)


def tensorboard_visual_deformation(name, dvf, grid_spacing, writer, step, **kwargs):
    if grid_spacing < 1:
        grid_spacing = 1
    figure = preview_3D_deformation(dvf, grid_spacing, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)


def tensorboard_visual_dvf(name, dvf, writer, step):
    figure = preview_3D_vector_field(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)


def tensorboard_visual_det(name, det, writer, step, **kwargs):
    figure = preview_image(det, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)


def tensorboard_visual_RGB_dvf(name, dvf, writer, step):
    figure = RGB_dvf(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)


def tensorboard_visual_warp_grid(name, warp_grid, writer, step, **kwargs):
    figure = comput_fig(warp_grid)
    # figure = preview_image(warp_grid, cmap='gray', **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)


def tensorboard_visual_deformation_2(name, dvf, writer, step):
    figure = PlotGrid_3d(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    writer.add_figure(name, figure, step)
