import math
import os
import shutil

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from loguru import logger as loguru_logger

def sitk_read(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img


def sitk_write(img, path):
    sitk.WriteImage(sitk.GetImageFromArray(img), path)


def clear_dir(path:str):
    if not os.path.exists(path):
        return
    
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除{file_path}时出错：{e}")


def slice_3d_img(img_path, axis, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    
    if axis == 1:
        img = img.transpose(1,0,2)
    elif axis == 2:
        img = img.transpose(2,0,1)
    
    # img_name = os.path.basename(img_path).split('.')[0] + ".png"
    # save_path = os.path.join(save_dir, img_name)
    for i, sl in enumerate(img):
        img_name = os.path.basename(img_path).split('.')[0] + f"_{i}.png"
        save_path = os.path.join(save_dir, img_name)
        sl = sitk.GetImageFromArray(sl)
        sitk.WriteImage(sl, save_path)
    
def cal_MSE(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    assert img1.shape == img2.shape
    
    size = math.prod(img1.shape)
    
    return np.sum((img1 - img2) ** 2) / size

def block_NCC(block1, block2):
    b1 = np.array(block1).astype(np.float32)
    b2 = np.array(block2).astype(np.float32)
    
    cc = np.sum((b1 - np.mean(b1)) * (b2 - np.mean(b2)))
    sta_dev = np.sqrt(np.sum(b1 ** 2) * np.sum(b2 ** 2)) + np.finfo(float).eps
    
    return cc / sta_dev

def cal_LNCC(img1, img2, block_size=(8, 8)):
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    assert img1.shape == img2.shape
    
    if block_size is None:
        return -block_NCC(img1,img2)
    
    h, w = img1.shape
    block_h, block_w = block_size
    n_h = h // block_h
    n_w = w // block_w
    
    ncc_map = np.zeros((n_h, n_w))

    for i in range(n_h):
        for j in range(n_w):
            block1 = img1[i * block_h:(i + 1) * block_h, j * block_w: (j + 1) * block_w]
            block2 = img2[i * block_h:(i + 1) * block_h, j * block_w: (j + 1) * block_w]
            ncc_map[i, j] = block_NCC(block1, block2)
    
    return -np.mean(ncc_map)

def compute_joint_histogram(image1, image2, bins=256):
    """计算联合直方图"""
    hist_2d, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    return hist_2d

def cal_MI(image1, image2, bins=256):
    """计算两张图像之间的互信息"""
    # 计算联合直方图
    joint_hist = compute_joint_histogram(image1, image2, bins)
    
    # 将联合直方图归一化为联合概率分布
    joint_prob = joint_hist / np.sum(joint_hist)
    

    # 计算边缘概率分布
    p1 = np.sum(joint_prob, axis=1, keepdims=True).repeat(bins, axis=1)
    p2 = np.sum(joint_prob, axis=0, keepdims=True).repeat(bins, axis=0)
    # 计算互信息
    non_zero = np.argwhere(joint_prob > 0)
    joint_prob_non_zero = joint_prob[non_zero]
    p1_non_zero = p1[non_zero]
    p2_non_zero = p2[non_zero]
    
    mutual_info = np.sum(joint_prob_non_zero * np.log(joint_prob_non_zero / (p1_non_zero * p2_non_zero)))
    
    return mutual_info

@loguru_logger.catch
def get_slice(image, normal, point, slice_shape):
    """
    获得三维图像上给定平面的切面
    """

    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    point = np.array(point)

    u = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(u) == 0:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    width, height = slice_shape
    uu, vv = np.meshgrid(np.linspace(-width // 2, width // 2, width),
                         np.linspace(-height // 2, height // 2, height))

    # 获取平面上坐标    
    slice_points = point[:, None, None] + uu[None, :, :] * u[:, None, None] + vv[None, :, :] * v[:, None, None]

    x, y, z = slice_points
    coords = np.array([x, y, z])

    # 插值
    if len(image.shape) == 3:
        slice_image = map_coordinates(image, coords, order=0, mode='nearest')
    else:
        slice_image = np.zeros((height, width, image.shape[-1])).astype(image.dtype)
        for i in range(image.shape[-1]):
            slice_image[..., i] = map_coordinates(image[..., i], coords, order=0, mode='nearest')
            
    return slice_image

def rotation_matrix(axis, theta):

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def generate_nearby_normals(base_normal, angle_range_degrees, step_degrees):
    base_normal = np.array(base_normal)
    angle_range_radians = np.radians(angle_range_degrees)
    step_radians = np.radians(step_degrees)
    
    normals = []
    angles = np.arange(-angle_range_radians, angle_range_radians + step_radians, step_radians)
    record_angles = []
    for theta_x in angles:
        for theta_y in angles:
            R_x = rotation_matrix([1, 0, 0], theta_x)
            rotated_x = R_x @ base_normal
            
            R_z = rotation_matrix([0, 1, 0], theta_y)
            rotated_normal = R_z @ rotated_x
            
            normals.append(rotated_normal)
            record_angles.append((theta_x, theta_y))
    
    return normals, record_angles


def img_norm(img):
    mx = img.max()
    mn = img.min()

    img = (img - mn) / (mx - mn) * 255
    return img.astype(np.uint8)
def adjust_contrast(image, level, window):
    """
    调整图像的对比度和亮度。
    
    level: 对比度中心
    window: 对比度宽度
    
    """
    # 计算最低和最高灰度值
    min_val = level - window / 2
    max_val = level + window / 2
    
    normalized_img = (image - min_val) / (max_val - min_val)
    
    normalized_img = np.clip(normalized_img, 0, 1)
    
    adjusted_img = (normalized_img * 255).astype(np.uint8)
    
    return adjusted_img