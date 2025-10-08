import os
import cv2
import sys
import torch
import random
import shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import matplotlib.colors as mcolors

def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)


def equalize_intensity_positive(img):
    mask = img > 0
    positive_vals = img[mask]
    hist, bins = np.histogram(positive_vals, bins=256, density=True)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    eq_positive = np.interp(positive_vals, bins[:-1], cdf_normalized)
    out = img.copy()
    out[mask] = eq_positive
    return out

def heatmap(save_img_path, edge_path, save_path, template_path, alpha):
    '''
    :param img_path: Segmentation image (axon or soma)
    :param edge_path: Edge of atlas, used as background lines
    :param save_path: Path to save heatmap
    :param alpha: Alpha of the heatmap
    :return: None
    '''
    img = read_tiff_stack(save_img_path)
    edge = read_tiff_stack(edge_path)
    template = read_tiff_stack(template_path)
    
    # coronal
    # template = template.transpose(1, 0, 2)
    
    # 新建热力图
    heatimg = np.zeros(img.shape)
    heatimg = np.array([heatimg, heatimg, heatimg]).transpose((1, 2, 3, 0))
    edge = np.array([edge, edge, edge]).transpose((1, 2, 3, 0))
    
    radiation_matrix = np.zeros((11, 11, 11))
    radiation_matrix[1:10, 1:10, 1:10] = 1
    radiation_matrix[4:7, 4:7, 4:7] = 2
    radiation_matrix[5, 5, 5] = 3
    
    radiation_matrix = radiation_matrix[np.newaxis, np.newaxis, ...]
    
    # no pooling
    conv = nn.Conv3d(1, 1, 11, 1, padding=5, bias=False)
    conv.weight = nn.Parameter(torch.Tensor(radiation_matrix), requires_grad=False)
    img_out = conv(torch.Tensor(img.astype(np.float32)[np.newaxis, np.newaxis, ...]))
    
    # pooling
    # conv = nn.Conv3d(1, 1, 11, 1, padding=5, bias=False)
    # conv.weight = nn.Parameter(torch.Tensor(radiation_matrix), requires_grad=False)
    # pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
    # imgpooling = pool(torch.Tensor(img.astype(np.float32)[np.newaxis, np.newaxis, ...]))
    # img_out = conv(imgpooling)
    
    
    img_out_heat = img_out.detach().numpy().squeeze()
    img_out_heat[template == 0] = 0
    img_out_heat_normed = (img_out_heat - img_out_heat.min()) / (img_out_heat.max() - img_out_heat.min())
    
    # axon
    # img_out_heat_eq = equalize_intensity_positive(img_out_heat_normed)
    
    # soma
    mask = img_out_heat_normed < 1e-2
    img_out_heat_gamma = np.power(img_out_heat_normed, 0.5)
    img_out_heat_gamma[mask] = img_out_heat_normed[mask]
    img_out_heat_eq = img_out_heat_gamma
    

    img_out_heat_eq *= alpha
    
    my_viridis2 = mcolors.LinearSegmentedColormap.from_list(
    "my_viridis",
        # soma
        ["#000000",  "#00009E",  "#367D9E", "#fde724", "#fb552e", "#fb552e", "#DE0000", "#DE0000", "#b90000", "#b90000"]
        
        # axon
        # ["#430154", "#0000FF",  "#30678d", "#72cf55", "#fde724", "#b90000"]
    )
    
    # colormap
    rgb = my_viridis2(img_out_heat_eq)[..., :3]  # shape: (h, w, 3)
    
    heatimg = (rgb * 255).astype(np.uint8)
    
    template = np.array([template, template, template]).transpose((1, 2, 3, 0))
    
    heatimg[template == 0] = 0
    heatimg[edge != 0] = edge[edge != 0]
    
    heatimg[heatimg < 0] = 0
    heatimg[heatimg > 255] = 255
    
    tifffile.imwrite(save_path, np.uint8(heatimg))

# ------------------------------ one brain ---------------------------------
# img_path = r"P28_Brain1_axon_final.tiff"
# edge_path = r"P28_edge.tiff"
# save_path = r"P28_Brain1_axon_final_heatmap.tiff"
# template_path = r"P28_template.tiff"
# 
# alpha = 1
# 
# heatmap(img_path, edge_path, save_path, template_path, alpha)

# ---------------------------- avg brain -------------------------------
img_path_root = r"C:\Users\test\Desktop\visual\processing\reg\new_final_resize\collected_symmetry"
edge_path_root = r'C:\Users\test\Desktop\visual\brainatlas_v4'
save_path_root = r"C:\Users\test\Desktop\visual\processing\reg\new_final_resize\collected_symmetry\heatmap"

stages = ["P0", "P4", "P7", "P10", "P14", "P21", "P28"]

type = 'axon_sk'

alpha = 1

stage_dict = {}
for stage in stages:
    stage_list = []
    for img in sorted(os.listdir(img_path_root)):
        if img.endswith(".tif") and type in img and img.split('_')[0] == stage:
            stage_list.append(img)
    stage_dict[stage] = stage_list

print(stage_dict)

for stage in stages:
    print("{} : {}".format(stage, len(stage_dict[stage])))

for stage in stages:
    edge_path = os.path.join(edge_path_root, stage) + '\\' + stage + '_edge_resize.tiff'
    template_path = os.path.join(edge_path_root, stage) + '\\' + stage + '_template.tiff'
    anno_path = os.path.join(edge_path_root, stage) + '\\' + stage + '_anno_resize.tiff'
    save_image_path = os.path.join(save_path_root, stage + '_' + type + '_avg_image.tiff')
    save_heatmap_path = os.path.join(save_path_root, stage + '_' + type + '_avg_heatmap.tiff')
    
    anno = read_tiff_stack(anno_path)
    regin_pixel = np.isin(anno, [672, 56, 754])

    avg_image = None
    brain_num = len(stage_dict[stage])
    for img in tqdm(sorted(os.listdir(img_path_root))):
        if img.endswith(".tif") and type in img and img.split('_')[0] == stage:
            print(img)
            one_img_path = os.path.join(img_path_root, img)
            if avg_image is None:
                avg_image = read_tiff_stack(one_img_path) / brain_num
            else:
                avg_image += read_tiff_stack(one_img_path) / brain_num

    avg_image = avg_image.astype(np.uint8)
    avg_image[regin_pixel] = 0
    # horizontal
    tifffile.imwrite(save_image_path, avg_image)
    # coronal
    # tifffile.imwrite(save_image_path, avg_image.transpose(1, 0, 2))   

    heatmap(save_image_path, edge_path, save_heatmap_path, template_path, alpha)