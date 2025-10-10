import json
from PIL import Image
import os
from os.path import join
import SimpleITK as sitk
import numpy as np
import pandas as pd
import tifffile
import skimage.io as io
from tqdm import tqdm
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
import skimage.io as io
from skimage.feature import hog
from skimage import exposure
import torch.nn.functional as nnf


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def collect_id_ytw_children(region, result, parent_name=None):
    """Recursively collect id_ytw values for all children of a region with st_level=5."""
    # Check if the current region has st_level = 5
    # if region.get('st_level') == 5:
    #     parent_name = region.get('acronym')
    #     if parent_name not in result:
    #         result[parent_name] = []

    # if region.get('acronym') in ["ASO", "ACA", "ORB", "MOB", "NLOT", "BMA", "CP", "ACB", "FS", "OT", "LS",
    #                              "CEA", "IA", "MEA", "GPi", "BST", "RE", "LGv", "LH", "PVH", "ARH", "ADP", "DMH", "PD",
    #                              "PS", "SCH", "SUM", "PH", "LPO", "STN", "ZI", "ME", "SNr", "VTA", "RR", "PAG", "SNc",
    #                              "IF", "SOC", "LC", "NTS", "LRN", "MDRNd", "PGRNl", "PRNc", "PRNr"]:

    if region.get('acronym') in ["Isocortex"]:
        parent_name = region.get('acronym')
        if parent_name not in result:
            result[parent_name] = []

    # Add the current region's id_ytw to the parent's list if parent_id_ytw exists
    if parent_name is not None:
        result[parent_name].append(region.get('id_ytw'))

    # Recurse through children
    for child in region.get('children', []):
        collect_id_ytw_children(child, result, parent_name)


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

def cal_soma_num_with_mask(image, mask):
    soma_number = np.sum(image[mask>0]) / 255

    return int(soma_number)


def main(mask_path, seg_path, analysis_excel_path):
    # Load the JSON data
    file_path = 'add_id_ytw.json'
    data = load_json(file_path)

    # Initialize result dictionary
    result = {}

    # Start recursive traversal
    collect_id_ytw_children(data, result)

    print(result)

    image = read_tiff_stack(seg_path)
    mask = read_tiff_stack(mask_path)
    print('load image and mask succsfully!!!')

    region_soma_data = []  # Store (region, soma number)

    for parent_name, child_id_ytw_list in result.items():
        print(parent_name)

        regin_pixel = np.isin(mask, child_id_ytw_list).astype(np.uint8)
        soma_number = cal_soma_num_with_mask(image, regin_pixel)

        print("soma num: {}".format(soma_number))

        region_soma_data.append((parent_name, soma_number))

    df = pd.DataFrame(region_soma_data, columns=["Brain region", "Soma number"])
    df.to_excel(analysis_excel_path, index=False)
    print(f"Saved result to {analysis_excel_path}")


if __name__ == "__main__":
    # mask_path = "/media/root/18TB_HDD/hzq/1_whole_brain_count_20250529/anno_upsample/P0_Brain1_warped_anno_upsample.tiff"
    # seg_path = "/media/root/18TB_HDD/hzq/1_whole_brain_count_20250529/whole_brain_center/seg_P0_Brain1_new_soma_24813_hog_union_post.tiff"
    # main(mask_path, seg_path)

    cfg_path = "add_id_ytw.json"
    base = r"/media/root/18TB_HDD/hzq/1_whole_brain_count_20250529/whole_brain_center1/"
    brain_list = [brain for brain in os.listdir(base) if brain.endswith('.tiff')]
    print(brain_list)

    for brain in brain_list:
        brain_name = brain.split('_new')[0][4:]
        print(brain_name)
        if brain_name not in ['P28_Brain1', 'P28_Brain3', 'P21_Brain5', 'P21_Brain7']:
            mask_path = "/media/root/18TB_HDD/hzq/1_whole_brain_count_20250529/anno_upsample/" + brain_name + "_warped_anno_upsample.tiff"
        else:
            mask_path = "/mnt/18TB_HDD2/lpq/TH_647_anno_upsampled/" + brain_name + "_warped_anno_upsample.tiff"

        analysis_excel_path = "/media/root/18TB_HDD/hzq/1_whole_brain_count_20250529/analysis/" + brain_name + "_soma_center_number_special_brain_region_Isocortex.xlsx"

        if os.path.exists(analysis_excel_path):
            continue

        seg_path = os.path.join(base, brain)
        main(mask_path, seg_path, analysis_excel_path)
