import os

import SimpleITK as sitk
import matplotlib.pyplot as plt

from util.visual.visual_image import preview_image
from util.visual.visual_registration import preview_3D_deformation, preview_3D_vector_field, RGB_dvf, PlotGrid_3d, \
    comput_fig


def save_slice(outdir, img_list, name_list, cmap='gray', figsize=(18, 18)):
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, 'slice.png')
    D, H, W = img_list[0].shape

    imgs = []
    imgs.append([img[D // 2, :, :, ] for img in img_list])
    imgs.append([img[:, H // 2, :] for img in img_list])
    imgs.append([img[:, :, W // 2] for img in img_list])
    fig, axs = plt.subplots(3, len(imgs), figsize=figsize)
    for axs_, ims_ in zip(axs, imgs):
        for ax, im in zip(axs_, ims_):
            ax.axis('off')
            ax.imshow(im, origin='lower', cmap=cmap)
    title = ','.join(name_list)
    plt.suptitle(title, fontsize=25)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


# def save_flow(savename, I_img, header=None, affine=None):
#     if header is None or affine is None:
#         affine = np.diag([1, 1, 1, 1])
#         new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
#     else:
#         new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)
#
#     nib.save(new_img, savename)


def write_image(outdir, name, img, type):
    os.makedirs(outdir, exist_ok=True)
    if len(type) > 0:
        name = name + '_' + type
    data_type = 'float'
    if type == 'label':
        data_type = 'uint8'
    img = img.cpu().numpy().astype(data_type)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, os.path.join(outdir, name + '.nii.gz'))


def save_image_figure(outdir, name, image, cmap='gray', **kwargs):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_image(image, cmap=cmap, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_deformation_figure(outdir, name, dvf, grid_spacing, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    if grid_spacing < 1:
        grid_spacing = 1
    figure = preview_3D_deformation(dvf, grid_spacing, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_dvf_figure(outdir, name, dvf):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_3D_vector_field(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_warp_grid_figure(outdir, name, warp_grid, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    figure = comput_fig(warp_grid)
    # figure = preview_image(warp_grid, cmap='gray', **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_det_figure(outdir, name, det, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_image(det, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_RGB_dvf_figure(outdir, name, dvf):
    os.makedirs(outdir, exist_ok=True)
    figure = RGB_dvf(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()


def save_RGB_deformation_2_figure(outdir, name, dvf):
    os.makedirs(outdir, exist_ok=True)
    figure = PlotGrid_3d(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
    plt.close()
