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

input_base = '/media/root/SSD/lpq/Tph2_axon_segmentations/to_thresh'
output_base = '/media/root/SSD/lpq/Tph2_axon_segmentations/threshed'

threshold_value = 125
for folder in sorted(os.listdir(input_base)):
    input_folder = os.path.join(input_base, folder)
    output_folder = os.path.join(output_base, folder)

    if os.path.exists(output_folder) and len(os.listdir(input_folder)) == len(os.listdir(output_folder)):
        continue

    else:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹A中的所有文件名
    file_names = [f for f in os.listdir(input_folder)]

    # 初始化进度条
    print("Thresholding TIFF files in {}".format(folder))

    for file_name in tqdm(file_names, desc="Processing"):
        # 构造文件路径
        file_path = os.path.join(input_folder, file_name)

        # 读取TIFF文件
        image = sitk.ReadImage(file_path)

        # 将图像转换为数组
        array_image = sitk.GetArrayFromImage(image)

        # A - B 操作
        result_final = (array_image >= threshold_value) * 255

        # 将结果转换回SimpleITK图像
        output_image = sitk.GetImageFromArray(result_final.astype('uint8'))
        output_image.SetSpacing(image.GetSpacing())
        output_image.SetOrigin(image.GetOrigin())
        output_image.SetDirection(image.GetDirection())

        # 保存结果图像c
        output_path = os.path.join(output_folder, file_name)
        sitk.WriteImage(output_image, output_path)

