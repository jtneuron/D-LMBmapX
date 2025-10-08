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
from vhog3d_GPU import hog3d_GPU
import myutils
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import tifffile
import argparse

join = os.path.join
train_patient_names = []
test_patient_names = []

def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out):
    img = imread(img_file)

    img_itk = sitk.GetImageFromArray(img)

    sitk.WriteImage(img_itk, join(img_out_base + ".nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l)
        sitk.WriteImage(l_itk, anno_out)


def np_convert_to_nifti_train(data, label, img_out_base, anno_out_base, casename):
    data = data / 65535
    img_itk = sitk.GetImageFromArray(data.astype(np.float32))
    l_itk = sitk.GetImageFromArray(label.astype(np.float32))

    sitk.WriteImage(img_itk, img_out_base + ".nii.gz")
    sitk.WriteImage(l_itk, anno_out_base + '.nii.gz')
    train_patient_names.append(casename)


def np_convert_to_nifti_test(data, label, img_out_base, anno_out_base, casename):
    data = data / 65535
    img_itk = sitk.GetImageFromArray(data.astype(np.float32))
    l_itk = sitk.GetImageFromArray(label.astype(np.float32))

    sitk.WriteImage(img_itk, img_out_base + ".nii.gz")
    sitk.WriteImage(l_itk, anno_out_base + '.nii.gz')
    test_patient_names.append(casename)


# base: volumes/labels/(labels_sk)/artifacts
# source: using for match_histograms
# output_dir: dir for all datasets used for training

def create_mae_data(base, output_dir, task_id, task_name, n_samples, input_dim):
    p = Pool(16)

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    data_path = join(base, 'train')
    volumes_folder_path = join(data_path, 'volumes')
    labels_folder_path = join(data_path, "labels")
    volumes_path = myutils.get_dir(volumes_folder_path)
    labels_path = myutils.get_dir(labels_folder_path)
    assert len(labels_path) == len(volumes_path)

    total_train_volumes = 0

    # generate train dataset
    with tqdm(total=len(volumes_path), desc=f'Train volume numbers') as pbar:
        for vpath, lpath in zip(volumes_path, labels_path):
            case = vpath.split(".")[0].split("-")[-1]
            casename = task_name + case

            volume = myutils.read_tiff_stack(vpath)
            label = myutils.read_tiff_stack(lpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue

            img_out_base = join(imagestr, casename)
            anno_out_base = join(labelstr, casename)

            np_convert_to_nifti_train(volume, label, img_out_base, anno_out_base, casename)

            total_train_volumes += 1
            pbar.update()

    print("{} data of train dataset finish.".format(total_train_volumes))

    # generate test dataset
    val_data_path = join(base, 'val')
    val_volumes_folder_path = join(val_data_path, 'volumes')
    val_labels_folder_path = join(val_data_path, "labels")
    val_volumes_path = myutils.get_dir(val_volumes_folder_path)
    val_labels_path = myutils.get_dir(val_labels_folder_path)
    assert len(val_volumes_path) == len(val_labels_path)

    total_test_volumes = 0

    with tqdm(total=len(val_volumes_path), desc=f'Test volume numbers') as pbar:
        for v_vpath, v_lpath in zip(val_volumes_path, val_labels_path):
            case = v_vpath.split(".")[0].split("-")[-1]
            casename = task_name + case

            volume = myutils.read_tiff_stack(v_vpath)
            label = myutils.read_tiff_stack(v_lpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue

            img_out_base = join(imagests, casename)
            anno_out_base = join(labelsts, casename)

            np_convert_to_nifti_test(volume, label, img_out_base, anno_out_base, casename)

            total_test_volumes += 1
            pbar.update()
    print("{} data of test dataset finish.".format(total_test_volumes))

    # write basic information of dataset to dataset.json, needed for nnUNet preprocessing
    json_dict = {'name': task_name, 'description': "", 'tensorImageSize': "4D", 'reference': "", 'licence': "",
                 'release': "0.0", 'modality': {
            "0": "MI",  # microscope image
        }, 'labels': {
            "0": "background",
            "1": "axon",
        }, 'numTraining': len(train_patient_names), 'numTest': len(test_patient_names),
                 'training': [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                              train_patient_names],
                 'test': [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                          test_patient_names]}

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
