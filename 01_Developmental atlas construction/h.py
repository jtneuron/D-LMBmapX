# 预处理--------------------------------------------------------------------------------------
import numpy as np
import SimpleITK as sitk
import os
import skimage

def process_my_data(AVGT):
    AVGT = AVGT.astype(np.float64)
    AVGT = np.rint((AVGT-np.min(AVGT)) / (np.max(AVGT)-np.min(AVGT)) * 255)
    AVGT = AVGT.astype(np.uint8)
    return AVGT

def N4_bias_correction(img):
    raw_img = img
    raw_img = process_my_data(raw_img).astype(np.float64)
    raw_img = sitk.GetImageFromArray(raw_img)
    correction_img = sitk.N4BiasFieldCorrection(raw_img, raw_img > 0)  # 需要图片是64位浮点类型
    correction_img = sitk.GetArrayFromImage(raw_img)
    return correction_img


def rescale_intensity(img):
    corr = img
    # RESCALE TO 8 BIT
    scale_limit = np.percentile(corr, (99.999))
    corr = skimage.exposure.rescale_intensity(corr, in_range=(0, scale_limit), out_range='uint8')
    img = np.copy(corr)

    # rescale intensity based on mean/median of tissue
    img_temp = np.copy(img)
    scale_thres = skimage.filters.threshold_otsu(img_temp)
    img_temp[img_temp < scale_thres] = 0
    nz_mean = np.mean(img_temp[img_temp > 0].flatten())
    print(nz_mean)
    scale_fact = 120 / nz_mean
    img = img * scale_fact
    print(np.max(img))
    img[img > 255] = 255
    img = img.astype('uint8')
    return img


def Clahe_3D(img):
    auto = img
    auto_temp_hor = np.copy(auto)
    auto_temp_cor = np.copy(auto)
    auto_temp_sag = np.copy(auto)

    for h in range(auto_temp_cor.shape[1]):
        temp_img = auto_temp_cor[:, h, :]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_cor[:, h, :] = clahe_im

    for h in range(auto_temp_hor.shape[0]):
        temp_img = auto_temp_hor[h, :, :]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_hor[h, :, :] = clahe_im

    for h in range(auto_temp_sag.shape[2]):
        temp_img = auto_temp_sag[:, :, h]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_sag[:, :, h] = clahe_im

    # combine angles
    clahe_all = np.zeros((auto_temp_sag.shape[0], auto_temp_sag.shape[1], auto_temp_sag.shape[2], 3))
    clahe_all[:, :, :, 0] = auto_temp_hor
    clahe_all[:, :, :, 1] = auto_temp_cor
    clahe_all[:, :, :, 2] = auto_temp_sag
    clahe_all_mean = np.mean(clahe_all, 3)

    # combine with original volume
    clahe_final = np.zeros((auto_temp_sag.shape[0], auto_temp_sag.shape[1], auto_temp_sag.shape[2], 2))
    clahe_final[:, :, :, 0] = clahe_all_mean
    clahe_final[:, :, :, 1] = auto

    clahe_final = np.mean(clahe_final, 3)
    clahe_final = np.uint8(clahe_final)
    return clahe_final


# img = sitk.GetArrayFromImage(sitk.ReadImage(file_name))
# img = np.transpose(img, (1, 2, 0))
# img = img[::-1, ::-1, ::-1]
# img1 = N4_bias_correction(img)
# img2 = rescale_intensity(img1)
# img3 = Clahe_3D(img)
# out_path = '/media/root/18TB_HDD/ljj/DateSet/488_10/p28_nifity/withOLF'
# sitk.WriteImage(sitk.GetImageFromArray(img2), os.path.join(out_path, os.path.basename(file_name).replace('.nii.gz', '_process.nii.gz')))

# 像素平均-------------------------------------------------------------------------------------
def average(path_list):
    brains = []
    for p in tqdm(path_list):
        img = sitk.ReadImage(p)
        img = sitk.GetArrayFromImage(img)
        brains.append(img)
    brains = np.array(brains)
    avg = np.mean(brains, axis=0)
    return np.uint8(avg)

# 标签的处理-----------------------------------------------------------------------------------
# label_dir = '/media/root/18TB_HDD/ljj/DateSet/p28_10/P28_pred_new'
# out_path = '/media/root/18TB_HDD/ljj/DateSet/488_10/p28_nifity/withOLF'
# brain_list = [b for b in os.listdir(out_path)]

# region_value = {
#     'BS': 1,
#     'CB': 2,
#     'CP': 3,
#     'CTX': 4,
#     'HPF': 5,
#     'OLF': 6
# }

# open = sitk.BinaryMorphologicalOpeningImageFilter()
# open.SetKernelRadius(1)
# close = sitk.BinaryMorphologicalClosingImageFilter()
# close.SetKernelRadius(5)

# for brain in brain_list:
#     label_img = np.zeros(shape=(800, 1140, 1320))
#     print("\n process: %s" % brain)
#     for i in os.listdir(label_dir):
#         label = i.split('_')[1]
#         print("\n extract: %s" % label)
#         for j in os.listdir(os.path.join(label_dir, i, 'label')):
#             if j[8:].replace('_' + label + '.nii', '') == brain:
#                 img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, i, 'label', j)))
#                 img = np.transpose(img, (1, 2, 0))
#                 img = img[::-1, ::-1, ::-1]
#                 label_img[img != 0] = region_value[label]
#         for j in os.listdir(os.path.join(label_dir, i, 'unlabel')):
#             if j[8:].replace('_' + label + '.nii', '') == brain:
#                 img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, i, 'unlabel', j)))
#                 img = np.transpose(img, (1, 2, 0))
#                 img = img[::-1, ::-1, ::-1]
#                 label_img[img != 0] = region_value[label]
#     label_img = sitk.GetImageFromArray(np.uint8(label_img))
#     label_img = open.Execute(label_img)
#     label_img = close.Execute(label_img)
#     sitk.WriteImage(label_img, os.path.join(out_path, brain, brain + '_process_label.nii.gz'))


