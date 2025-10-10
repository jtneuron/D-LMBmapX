import os
import tifffile
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    root = '/media/user/hdd1/liuhe/i2i_net/FGDM/output/translation/test/allen_dev_to_P14_avg/sample00_th_123/val'
    save = '/media/user/hdd1/liuhe/i2i_net/FGDM/output/translation/test/allen_dev_to_P14_avg/sample00_th_123/resize'
    os.makedirs(save, exist_ok=True)
    dir_list =  filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root))
    # dirs = ['Slc6a2_P14_20',
    #         'TH-P28_40',
    #         'TH-P28_50',
    #         'Tph2-P28_20',
    #         'Tph2-P28_40',
    #         'Tph2-P28_50',s
    #         # 'data_fvbex_brain8_m_re',
    #         # 'data_fvbex_brain6_m_re'
    #         ]  #
    dirs = sorted(list(dir_list))
    typ = '_translated'
    for d in dirs:
        imgs = sorted(os.listdir(os.path.join(root, d)), key=lambda x: int(str(x).split('_')[0]))  # _
        print(len(imgs))
        tiff = []

        for i in range(len(imgs)):
            # if typ in imgs[i]:
            im = np.array(Image.open(os.path.join(root, d, imgs[i])).convert('L'))
            im = cv2.resize(im, (256, 160), interpolation=cv2.INTER_CUBIC)
            print(os.path.join(root, d, imgs[i]))
            print(im.shape)
            # tiff.append(im)

            # tiff = tiff[::-1]
            im = np.array(im)
            print(im.shape)
            os.makedirs(os.path.join(save, d), exist_ok=True)
            tifffile.imwrite(os.path.join(save, d, imgs[i]), im.astype(np.uint8))


