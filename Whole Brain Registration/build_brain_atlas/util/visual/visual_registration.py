import copy
import math

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
from matplotlib.collections import LineCollection

from util.visual.visual_image import preview_image


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
        Note: vector_field spatial indices are swapped to match the conventions of imshow and quiver
    kwargs are passed to matplotlib plotting
    """
    # phi in the following line is the deformation mapping.
    # Note that we are swapping the spatial x and y when we evaluate vector_field;
    # the reason for this is that we want to match the the "matrix" or "image" style
    # conventions used by matplotlib imshow and quiver, where the axis used for "rows"
    # precedes the axis used for "columns"

    # phi = lambda pt: pt + vector_field[:, pt[1], pt[0]].numpy()  # deformation mapping

    phi = lambda pt: pt + vector_field[[1, 0], pt[1], pt[0]].numpy()  # deformation mapping

    # _, xmax, ymax = vector_field.shape
    _, ymax, xmax = vector_field.shape
    xvals = np.arange(0, xmax, grid_spacing)
    yvals = np.arange(0, ymax, grid_spacing)
    for x in xvals:
        pts = [phi(np.array([x, y])) for y in yvals]
        pts = np.array(pts)
        plt.plot(pts[:, 0], pts[:, 1], **kwargs)
    for y in yvals:
        pts = [phi(np.array([x, y])) for x in xvals]
        pts = np.array(pts)
        plt.plot(pts[:, 0], pts[:, 1], **kwargs)


def preview_3D_deformation(vector_field, grid_spacing, figsize=(18, 6), is_show=False, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    figure = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    if is_show:
        plt.show()
    return figure


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, figsize=(18, 6), is_show=False, downsampling=None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    figure = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    if is_show:
        plt.show()
    return figure


def RenderDVF(dvf, coef=20, thresh=1.):
    dvf = copy.deepcopy(dvf)
    dvf = np.abs(dvf)
    dvf = np.exp(-dvf / coef)
    dvf = dvf * thresh
    return dvf


def RGB_dvf(vector_field, is_show=False, figsize=(18, 6)):
    dvf = RenderDVF(vector_field)
    axis_order = tuple(range(1, vector_field.ndim)) + (0,)
    if torch.is_tensor(dvf):
        dvf = dvf.cpu().numpy()
    dvf = dvf.transpose(axis_order)
    figure = preview_image(dvf, is_show=is_show, figsize=figsize)
    return figure


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img)
    return grid_img


def comput_fig(img):
    x = img.shape[0] // 2
    img = img[(x - 8):(x + 8), :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def PlotFlow(pts, div=2, cmap="YlGnBu", norm=plt.Normalize(vmin=50, vmax=77)):
    pts = np.array(pts)
    segments = [(j / div, (j + 1) / div) for j in range(div)]
    lines = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for sa, sb in segments:
            pa = a * (1 - sa) + b * sa
            pb = a * (1 - sb) + b * sb

            lines.append((pa, pb))
    lc = LineCollection(np.array(lines), cmap=cmap, norm=norm, linewidth=1, alpha=1)
    plt.gca().add_collection(lc)


def PlotGrid_2d(deformation_2d):
    deformation_2d = deformation_2d.numpy()
    w, h = deformation_2d.shape[:-1]
    plt.axis((0, w, 0, h))
    plt.axis("off")
    for i in range(0, w, 1):
        PlotFlow([d for d in deformation_2d[i, :] if not math.isnan(d[0])])
    for i in range(0, h, 1):
        PlotFlow([d for d in deformation_2d[:, i] if not math.isnan(d[0])])


def PlotGrid_3d(dvf=None, is_show=False, figsize=(18, 6)):
    if dvf is None:
        dvf = np.concatenate([np.indices((128, 128)).transpose(1, 2, 0), np.ones((128, 128, 1)) * 64], axis=-1)

    size = dvf.shape[1:]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = grid.float()
    deformation_space = dvf + grid

    x, y, z = np.array(deformation_space.shape[1:]) // 2

    deformation_space = deformation_space.permute((1, 2, 3, 0)).contiguous()

    figure = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    PlotGrid_2d(deformation_space[x, :, :, [1, 2]])
    plt.subplot(1, 3, 2)
    PlotGrid_2d(deformation_space[:, y, :, [0, 2]])
    plt.subplot(1, 3, 3)
    PlotGrid_2d(deformation_space[:, :, z, [0, 1]])
    figure.tight_layout()
    if is_show:
        plt.show()
    return figure
