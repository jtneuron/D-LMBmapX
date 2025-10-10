import os
import cv2
import sys
import torch
import random
import shutil
import tifffile
import imageio
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from PIL import Image
from tqdm import tqdm
from scipy import misc
from libtiff import TIFF
import argparse


def save_resized_tiff_by_shape(root, target, shapes, mod='single'):
    images = []
    shape_x, shape_y, shape_z = shapes
    if mod == 'single':
        stack = TIFF.open(root, mode='r')
        for img in tqdm(list(stack.iter_images())):
            img = np.array(img).astype(np.uint16)
            img = cv2.resize(img, (shape_x, shape_y))
            images.append(img)
    elif mod == 'files':
        for i in tqdm(sorted(os.listdir(root))):
            img = Image.open(os.path.join(root, i))
            img = np.array(img).astype(np.uint16)
            img = cv2.resize(img, (shape_x, shape_y))
            images.append(img)
    images = np.array(images).transpose((2, 1, 0))
    _y, _z = images[0].shape
    res_img = np.array([cv2.resize(im, (shape_z, _y)) for im in images]).astype(np.uint16)
    tifffile.imwrite(target, res_img.transpose(2, 1, 0))


def main(args):
    # 文件夹路径
    folder_axon = args.folder_axon
    folder_soma = args.folder_soma
    output_folder = args.output_folder
    output_tiff = args.output_folder + '.tiff'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    if len(os.listdir(folder_soma)) != len(os.listdir(folder_axon)):
        print("soma and axon not match")
        return

    # 获取文件夹A中的所有文件名
    file_names = [f for f in os.listdir(folder_axon)]

    # 初始化进度条
    print("Processing TIFF files in {}".format(folder_axon.split('/')[-1]))
    for file_name in tqdm(file_names, desc="Processing"):
        # 构造文件路径
        file_axon_path = os.path.join(folder_axon, file_name)
#        file_soma_path = os.path.join(folder_soma, 'slice_0' + file_name)
        file_soma_path = os.path.join(folder_soma, file_name)


        # 读取TIFF文件
        image_axon = sitk.ReadImage(file_axon_path)
        image_soma = sitk.ReadImage(file_soma_path)

        # 将图像转换为数组
        array_axon = sitk.GetArrayFromImage(image_axon)
        array_soma = sitk.GetArrayFromImage(image_soma)

        # A - B 操作
        result_final = ((array_axon == 255) & (array_soma == 255)) * 0 + ((array_axon == 255) & (array_soma == 0)) * 255

        # 将结果转换回SimpleITK图像
        output_image = sitk.GetImageFromArray(result_final.astype('uint8'))
        output_image.SetSpacing(image_axon.GetSpacing())
        output_image.SetOrigin(image_axon.GetOrigin())
        output_image.SetDirection(image_axon.GetDirection())

        # 保存结果图像c
        output_path = os.path.join(output_folder, file_name)
        sitk.WriteImage(output_image, output_path)

    print("Processing complete. Results are saved in:", output_folder)
    print("Being resizing...")
    save_resized_tiff_by_shape(root=output_folder, target=output_tiff, shapes=[456, 528, 320], mod='files')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_axon', type=str, default=None, help='axon dir path')
    parser.add_argument('--folder_soma', type=str, default=None, help='soma dir path')
    parser.add_argument('--output_folder', type=str, default=None, help='output dir path')
    args = parser.parse_args()

    main(args)
