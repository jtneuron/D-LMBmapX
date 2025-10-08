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


def main(args):
    # ---------------------------- One soma folders ---------------------------------
    # 文件夹路径
    folder_axon = args.folder_axon
    folder_soma1 = args.folder_soma1
    output_folder = args.output_folder

    threshold_value = 125

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)


    # 获取文件夹A中的所有文件名
    file_names_axon = [f for f in os.listdir(folder_axon)]
    file_names_soma1 = [f for f in os.listdir(folder_soma1)]

    len_dis1 = len(file_names_axon) - len(file_names_soma1)

    print("Processing TIFF files in {}".format(folder_axon.split('/')[-1]))
    for i in tqdm(range(len(file_names_axon)), desc='Processing:'):
        file_name = f"{i:04d}.tiff"
        file_axon_path = os.path.join(folder_axon, file_name)
        file_soma_path1 = os.path.join(folder_soma1, file_name)

        image_axon = sitk.ReadImage(file_axon_path)
        array_axon = sitk.GetArrayFromImage(image_axon)
        array_axon = (array_axon >= threshold_value) * 255

        if i < len_dis1:
            result_final = array_axon

        elif i >= len_dis1:
            image_soma1 = sitk.ReadImage(file_soma_path1)

            array_soma1 = sitk.GetArrayFromImage(image_soma1)
            array_soma1 = (array_soma1 >= threshold_value) * 255

            result_final = ((array_axon == 255) | (array_soma1 == 255)) * 255

        output_image = sitk.GetImageFromArray(result_final.astype('uint8'))
        output_image.SetSpacing(image_axon.GetSpacing())
        output_image.SetOrigin(image_axon.GetOrigin())
        output_image.SetDirection(image_axon.GetDirection())

        output_path = os.path.join(output_folder, file_name)
        sitk.WriteImage(output_image, output_path)

    print("Processing complete. Results are saved in:", output_folder)


    # ------------------------------------ Two soma folders ---------------------------------
    # # 文件夹路径
    # folder_axon = args.folder_axon
    # folder_soma1 = args.folder_soma1
    # folder_soma2 = args.folder_soma2
    # output_folder = args.output_folder
    #
    # threshold_value = 125
    #
    # # 确保输出文件夹存在
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    #
    # # 获取文件夹A中的所有文件名
    # file_names_axon = [f for f in os.listdir(folder_axon)]
    # file_names_soma1 = [f for f in os.listdir(folder_soma1)]
    # file_names_soma2 = [f for f in os.listdir(folder_soma2)]
    #
    # len_dis1 = len(file_names_axon) - len(file_names_soma1)
    # len_dis2 = len(file_names_axon) - len(file_names_soma2)
    #
    # print("Processing TIFF files in {}".format(folder_axon.split('/')[-1]))
    # for i in tqdm(range(len(file_names_axon)), desc='Processing:'):
    #     file_name = f"{i:04d}.tiff"
    #     file_axon_path = os.path.join(folder_axon, file_name)
    #     file_soma_path1 = os.path.join(folder_soma1, file_name)
    #     file_soma_path2 = os.path.join(folder_soma2, file_name)
    #
    #     image_axon = sitk.ReadImage(file_axon_path)
    #     array_axon = sitk.GetArrayFromImage(image_axon)
    #     # array_axon = (array_axon >= threshold_value) * 255
    #
    #     if i < len_dis1:
    #         result_final = array_axon
    #
    #     elif i >= len_dis1 and i < len_dis2:
    #         image_soma1 = sitk.ReadImage(file_soma_path1)
    #
    #         array_soma1 = sitk.GetArrayFromImage(image_soma1)
    #         array_soma1 = (array_soma1 >= threshold_value) * 255
    #
    #         result_final = ((array_axon == 255) | (array_soma1 == 255)) * 255
    #
    #     else:
    #         image_soma1 = sitk.ReadImage(file_soma_path1)
    #         image_soma2 = sitk.ReadImage(file_soma_path2)
    #
    #         array_soma1 = sitk.GetArrayFromImage(image_soma1)
    #         array_soma2 = sitk.GetArrayFromImage(image_soma2)
    #
    #         array_soma1 = (array_soma1 >= threshold_value) * 255
    #         array_soma2 = (array_soma2 >= threshold_value) * 255
    #
    #         result_final = ((array_axon == 255) | (array_soma1 == 255) | (array_soma2 == 255)) * 255
    #
    #     output_image = sitk.GetImageFromArray(result_final.astype('uint8'))
    #     output_image.SetSpacing(image_axon.GetSpacing())
    #     output_image.SetOrigin(image_axon.GetOrigin())
    #     output_image.SetDirection(image_axon.GetDirection())
    #
    #     output_path = os.path.join(output_folder, file_name)
    #     sitk.WriteImage(output_image, output_path)
    #
    # print("Processing complete. Results are saved in:", output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_axon', type=str, default=None, help='axon dir path')
    parser.add_argument('--folder_soma1', type=str, default=None, help='soma dir path')
    parser.add_argument('--folder_soma2', type=str, default=None, help='soma dir path')
    parser.add_argument('--output_folder', type=str, default=None, help='output dir path')
    args = parser.parse_args()

    main(args)
