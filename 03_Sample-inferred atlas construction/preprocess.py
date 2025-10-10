from h import N4_bias_correction, rescale_intensity, Clahe_3D, average
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm



# file_path = f"/media/user/phz/data/488/P0_1320_800_1140/P0_resize"
# out_path = f"/media/user/phz/data/488/P0_temp/P0_preprocess"

# for brain_name in tqdm(os.listdir(file_path), desc="预处理"):
#     file_name = os.path.join(file_path, brain_name)
#     img = sitk.GetArrayFromImage(sitk.ReadImage(file_name))
#     img = np.transpose(img, (1, 2, 0))

#     img = img[::-1, ::-1, ::-1]
#     img1 = N4_bias_correction(img)
#     img2 = rescale_intensity(img1)
#     img3 = Clahe_3D(img2)

    
#     os.makedirs(os.path.join(out_path, brain_name.split('.')[0]), exist_ok=True)
#     sitk.WriteImage(sitk.GetImageFromArray(img3), os.path.join(out_path, brain_name.split('.')[0], brain_name.replace('.tiff', '_process.nii.gz')))

# # 像素平均-------------------------------------------------------------------------------------
# def average(path_list):
#     brains = []
#     for p in tqdm(path_list):
#         p = os.path.join(p, os.path.basename(p)+"_process.nii.gz")
#         img = sitk.ReadImage(p)
#         img = sitk.GetArrayFromImage(img)
#         brains.append(img)
#     brains = np.array(brains)
#     avg = np.mean(brains, axis=0)
#     return np.uint8(avg)

# path = f"/media/user/phz/data/488/P0_temp/P0_preprocess"
# path_list = [os.path.join(path, x) for x in os.listdir(path)]
# ave = average(path_list)
# pre_ave_path = f"/media/user/phz/data/488/P0_temp/ave"
# os.makedirs(pre_ave_path, exist_ok=True)
# sitk.WriteImage(sitk.GetImageFromArray(ave), os.path.join(pre_ave_path, "ave.nii.gz"))



exit(0)
# 标签的处理-----------------------------------------------------------------------------------
label_dir = '/media/user/phz/data/488/P0_1320_800_1140/P0_pred'
out_path = '/media/user/phz/data/488/P0_temp/P0_preprocess'
brain_list = [b for b in os.listdir(out_path)]

region_value = {
    'BS': 1,
    'CB': 2,
    'CP': 3,
    'CTX': 4,
    'HPF': 5,
    'OLF': 6
}

open = sitk.BinaryMorphologicalOpeningImageFilter()
open.SetKernelRadius(1)
close = sitk.BinaryMorphologicalClosingImageFilter()
close.SetKernelRadius(5)

for brain in tqdm(brain_list, desc="处理标签"):
    label_img = np.zeros(shape=(800, 1140, 1320))
    print("\n process: %s" % brain)
    for i in os.listdir(label_dir):
        label = i.split('_')[0]
        print("\n extract: %s" % label)
        for j in os.listdir(os.path.join(label_dir, i, 'label')):
            if j[8:].replace('_' + label + '.nii', '') == brain:
                img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, i, 'label', j)))
                img = np.transpose(img, (1, 2, 0))
                img = img[::-1, ::-1, ::-1]
                label_img[img != 0] = region_value[label]
        for j in os.listdir(os.path.join(label_dir, i, 'unlabel')):
            if j[8:].replace('_' + label + '.nii', '') == brain:
                img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, i, 'unlabel', j)))
                img = np.transpose(img, (1, 2, 0))
                img = img[::-1, ::-1, ::-1]
                label_img[img != 0] = region_value[label]
    label_img = sitk.GetImageFromArray(np.uint8(label_img))
    label_img = open.Execute(label_img)
    label_img = close.Execute(label_img)
    sitk.WriteImage(label_img, os.path.join(out_path, brain, brain + '_process_label.nii.gz'))




