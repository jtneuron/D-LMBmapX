import os
import tifffile
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    root = '/media/user/hdd1/liuhe/i2i_net/FGDM/ex_output/translation/ex07_segbranch_c/allen_to_P28_avg/sample00_image_size_256/val/translated'
    view = 'c'
    # view = 'a'
    # view = 's'

    save = os.path.join(os.path.dirname(root), "3d")
    os.makedirs(save, exist_ok=True)
    dir_list = filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root))
    # dirs = ['data_fvbex_brain1_m_re',
    #         'data_fvbex_brain2_m_re',
    #         'data_fvbex_brain3_m_re',
    #         'data_fvbex_brain4_m_re',
    #         'data_fvbex_brain5_m_re',
    #         'data_fvbex_brain7_m_re',
    #         'data_fvbex_brain8_m_re',
    #         'data_fvbex_brain6_m_re'
    #         ]  #
    dirs = sorted(list(dir_list))
    typ = '_translated'
    for d in dirs:
        imgs = sorted(os.listdir(os.path.join(root, d)), key=lambda x: int(str(x).split('.')[0]))  # _
        print(len(imgs))
        tiff = []

        for i in range(len(imgs)):
            # if typ in imgs[i]:
            im = np.array(Image.open(os.path.join(root, d, imgs[i])).convert('L'))
            h, w = im.shape[0], im.shape[1]

            if view == 'c':
                tmp = np.maximum(im[:, :int(w / 2)], cv2.flip(im[:, int(w / 2):], 1, dst=None))
                im = np.append(tmp, cv2.flip(tmp, 1, dst=None), axis=1)
                im = cv2.resize(im, (448, 320), interpolation=cv2.INTER_CUBIC)
            elif view == 'a':
                tmp = np.maximum(im[:int(h / 2), :], cv2.flip(im[int(h / 2):, :], 0, dst=None))
                im = np.append(tmp, cv2.flip(tmp, 0, dst=None), axis=0)
                im = cv2.resize(im, (512, 448), interpolation=cv2.INTER_CUBIC)  # a
            elif view == 's':
                im = cv2.resize(im, (512, 320), interpolation=cv2.INTER_CUBIC)  # s

            print(os.path.join(root, d, imgs[i]))
            print(im.shape)
            tiff.append(im)

        if view == 'c':
            tiff = tiff
            # tiff = tiff[::-1]
        elif view == 'a':
            tiff = np.transpose(tiff, (2, 0, 1))  # a
        elif view == 's':
            tiff = np.transpose(tiff, (2, 1, 0))
        tiff = np.array(tiff)
        print(tiff.shape)
        print(os.path.join(root, d + typ + '.tiff'))
        tifffile.imwrite(os.path.join(save, d + typ + '.tiff'), tiff.astype(np.uint8))
