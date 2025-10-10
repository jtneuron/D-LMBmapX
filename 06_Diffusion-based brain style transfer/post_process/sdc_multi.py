import os

import cv2
import numpy as np
from PIL import Image
from einops import rearrange


def get_max(concat_img, img_shape):
    concat_img = rearrange(np.array(concat_img), 'b x y -> x y b')
    new_img = np.array([])
    for row in concat_img:
        temp = np.max(row, axis=1).tolist()
        new_img = np.append(new_img, temp)
    new_img = np.reshape(new_img, img_shape)
    new_img = cv2.resize(new_img, (448, 320), interpolation=cv2.INTER_CUBIC)
    return new_img


def get_avg(concat_img, img_shape):
    concat_img = rearrange(np.array(concat_img), 'b x y -> x y b')
    new_img = np.array([])
    for row in concat_img:
        temp = np.average(row, axis=1).tolist()
        new_img = np.append(new_img, temp)
    new_img = np.reshape(new_img, img_shape)
    new_img = cv2.resize(new_img, (448, 320), interpolation=cv2.INTER_CUBIC)
    return new_img


def get_first(concat_img, img_shape):
    # concat_img = rearrange(np.array(concat_img), 'b x y -> x y b')
    # new_img = np.array([])
    # for row in concat_img:
    #     temp = row[0]  # np.average(row, axis=1).tolist()
    #     new_img = np.append(new_img, temp)
    new_img = concat_img[0]
    new_img = np.reshape(new_img, img_shape)
    new_img = cv2.resize(new_img, (448, 320), interpolation=cv2.INTER_CUBIC)
    return new_img


def get_second(concat_img, img_shape):
    # concat_img = rearrange(np.array(concat_img), 'b x y -> x y b')
    # new_img = np.array([])
    # for row in concat_img:
    #     temp = row[0]  # np.average(row, axis=1).tolist()
    #     new_img = np.append(new_img, temp)
    new_img = concat_img[1]
    new_img = np.reshape(new_img, img_shape)
    new_img = cv2.resize(new_img, (448, 320), interpolation=cv2.INTER_CUBIC)
    return new_img


def get_third(concat_img, img_shape):
    # concat_img = rearrange(np.array(concat_img), 'b x y -> x y b')
    # new_img = np.array([])
    # for row in concat_img:
    #     temp = row[0]  # np.average(row, axis=1).tolist()
    #     new_img = np.append(new_img, temp)
    new_img = concat_img[-1]
    new_img = np.reshape(new_img, img_shape)
    new_img = cv2.resize(new_img, (448, 320), interpolation=cv2.INTER_CUBIC)
    return new_img


if __name__ == '__main__':
    root = '/media/user/hdd1/liuhe/i2i_net/FGDM/output/translation/ex02_sdc/mri_to_lsfm/sample06/val'
    save_root = '/media/user/hdd1/liuhe/i2i_net/FGDM/output/translation/ex02_sdc/mri_to_lsfm/sample06/post_pro/third'
    os.makedirs(save_root, exist_ok=True)

    dirs = ['data_fvbex_brain1_m_re', 'data_fvbex_brain5_m_re', 'data_fvbex_brain6_m_re']  # sorted(list(dir_list))
    for d in dirs:
        os.makedirs(os.path.join(save_root, d), exist_ok=True)
        img_paths = sorted(os.listdir(os.path.join(root, d)), key=lambda x: int(str(x).split('_')[0]))

        pre_num = '1000'
        concat_img = []
        shape = ()
        for path in img_paths:
            img = np.array(Image.open(os.path.join(os.path.join(root, d), path)).convert('L'))
            num = os.path.basename(path).split('_')[0]
            shape = img.shape
            if num != pre_num and len(concat_img) != 0:
                print(len(concat_img))
                # new_img = get_max(concat_img, shape)
                # new_img = get_avg(concat_img, shape)
                # new_img = get_first(concat_img, shape)
                # new_img = get_second(concat_img, shape)
                new_img = get_third(concat_img, shape)

                print(os.path.join(save_root, d, pre_num + '.png'))
                cv2.imwrite(os.path.join(save_root, d, pre_num + '.png'), new_img)

                concat_img = []

            concat_img.append(img)
            pre_num = num

        new_img = get_avg(concat_img, shape)
        print(os.path.join(save_root, d, pre_num + '.png'))
        cv2.imwrite(os.path.join(save_root, d, pre_num + '.png'), new_img)
