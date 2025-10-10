import os
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from PIL import Image
import random
from batchgenerators.utilities.file_and_folder_operations import *
# from nnunet.paths import nnUNet_raw_data
# from nnunet.paths import preprocessing_output_dir
from skimage.io import imread
import pdb
from tqdm import tqdm
import myutils
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import tifffile
import argparse

join = os.path.join


def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out):
    img = imread(img_file)

    # img = gamma_correction(img, 1 / 4)
    img = auto_contrast_3d(img, percent=0.5)

    img_itk = sitk.GetImageFromArray(img)

    sitk.WriteImage(img_itk, join(img_out_base + ".nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l)
        sitk.WriteImage(l_itk, anno_out)


def np_convert_to_nifti(data, label, img_out_base, anno_out):
    img_itk = sitk.GetImageFromArray(data)
    sitk.WriteImage(img_itk, join(img_out_base + ".nii.gz"))

    if label is not None:
        l_itk = sitk.GetImageFromArray(label)
        sitk.WriteImage(l_itk, anno_out)


def auto_contrast_3d(volume, percent=1):
    """
    对3D图像进行自动对比度调整
    :param volume: 输入3D图像
    :param percent: 剪切百分比，用于忽略极端值
    :return: 对比度增强后的3D图像
    """
    # 计算指定百分比的低值和高值
    p_low, p_high = np.percentile(volume, (percent, 100 - percent))

    # 使用rescale_intensity重新映射像素值到0到65535范围
    volume_rescaled = exposure.rescale_intensity(volume, in_range=(p_low, p_high), out_range=(0, 65535))

    volume_rescaled = np.round(volume_rescaled).astype(np.uint16)

    return volume_rescaled


def gamma_correction(image, gamma):
    # 确保图像数据在0到1之间
    image = image / 65535
    # 应用伽马矫正
    image_corrected = np.power(image, gamma)
    # 将图像数据恢复到0到65535的范围
    image_corrected = image_corrected * 65535
    # 通常，我们希望输出的图像像素值为整数
    image_corrected = np.round(image_corrected).astype(np.uint16)
    return image_corrected


# base: volumes/labels/(labels_sk)/artifacts
# source: using for match_histograms
# output_dir: dir for all datasets used for training
def create_mae_data(base, output_dir, task_id, task_name, n_samples, input_dim):
    p = Pool(16)

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)

    train_patient_names = []
    test_patient_names = []

    volumes_folder_path = join(base, 'train')
    volumes_path = myutils.get_dir(volumes_folder_path)

    total_ori_volumes = 0

    # generate train dataset
    with tqdm(total=len(volumes_path), desc=f'Train volume numbers') as pbar:
        for vpath in volumes_path:
            case = vpath.split(".")[0].split("-")[-1]
            casename = task_name + case

            # casenames.append(casename)

            volume = myutils.read_tiff_stack(vpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue

            # GAMMA
            # volume = gamma_correction(volume, 1 / 4)

            # Auto contrast
            volume = auto_contrast_3d(volume, percent=0.5)

            img_out_base = join(imagestr, casename)

            label = None
            anno_out = None
            p.starmap_async(np_convert_to_nifti, ((volume, label, img_out_base, anno_out),))

            train_patient_names.append(casename)

            # datas_ori.append(volume)
            total_ori_volumes += 1
            pbar.update()

    print("{} data of train dataset finish.".format(total_ori_volumes))

    # generate test set
    val_volume_path = join(base, "val")
    vpaths = os.listdir(val_volume_path)

    with tqdm(total=len(vpaths), desc=f'Val volume numbers') as pbar:
        for i, vpath in enumerate(vpaths):
            case = str(vpath).split(".")[0].split("-")[-1]
            volume = join(val_volume_path, "volume-" + case + ".tiff")
            label = None
            casename = task_name + case
            img_out_base = join(imagests, casename)
            anno_out = None
            p.starmap_async(load_tiff_convert_to_nifti, ((volume, label, img_out_base, anno_out),))

            test_patient_names.append(casename)
            pbar.update()

    # write basic information of dataset to dataset.json, needed for nnUNet preprocessing
    json_dict = {'name': task_name, 'description': "", 'tensorImageSize': "4D", 'reference': "", 'licence': "",
                 'release': "0.0", 'modality': {
            "0": "MI",  # microscope image
        }, 'labels': {
            "0": "background",
            "1": "axon",
        }, 'numTraining': len(train_patient_names), 'numTest': len(test_patient_names),
                 'training': [{'image': "./imagesTr/%s.nii.gz" % i} for i in
                              train_patient_names],
                 'test': [{'image': "./imagesTs/%s.nii.gz" % i} for i in test_patient_names]}

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    p.close()
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default=None, help='train data path')
    parser.add_argument('--output_dir', type=str, default=None, help='(if needed)data used for histogram match')
    parser.add_argument('--task_id', type=int, default=501,
                        help='task id should be unique(better >200 to avoid conflict)')
    parser.add_argument('--task_name', type=str, default='Axon_validate_BS', help='task name')

    args = parser.parse_args()

    n_samples = 1
    input_dim = 128
    create_mae_data(args.base, args.output_dir, args.task_id, args.task_name, n_samples, input_dim)
