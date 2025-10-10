import os
import pdb
import time
import shutil
import tifffile
import argparse
import numpy as np
from PIL import Image
import random
from multiprocessing import Pool
from tqdm import tqdm

'''
    root：脑子所有切片文件夹
    target：cubes保存路径
    width/thickness：切割cube的尺寸
    indx_start：cube命名起始序号
    sp_x：脑子采样区间起始点的x
    sp_y：脑子采样区间起始点的y
    sp_z：脑子采样区间起始点的z
    n_cubes：需要切分的cube的数量
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CubeCropper', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=None, help='source directory of the brain slices')
    parser.add_argument('--target', type=str, default=None, help='target directory of the cropped volumes')
    parser.add_argument('--width', type=int, default=128, help='width of a cube')
    parser.add_argument('--thickness', type=int, default=128, help='thickness of a cube')
    parser.add_argument('--indx_start', type=int, default=None, help='starting index of the output files')
    parser.add_argument('--sp_x', type=int, default=0, help='starting point x')
    parser.add_argument('--sp_y', type=int, default=0, help='starting point y')
    parser.add_argument('--sp_z', type=int, default=0, help='starting point z')
    parser.add_argument('--ep_x', type=int, default=None, help='ending point x')
    parser.add_argument('--ep_y', type=int, default=None, help='ending point y')
    parser.add_argument('--ep_z', type=int, default=None, help='ending point z')
    parser.add_argument('--n_cubes', type=int, default=None, help='total cubes for once sampling')
    parser.add_argument('--batch_size', type=int, default=None, help='num of cubes for one z-axis batch')

    args = parser.parse_args()


    root = args.root
    target = args.target
    width = args.width
    thickness = args.thickness
    indx_start = args.indx_start
    sp_x = args.sp_x
    sp_y = args.sp_y
    sp_z = args.sp_z
    ep_x = args.ep_x
    ep_y = args.ep_y
    ep_z = args.ep_z
    n_cubes = args.n_cubes
    batch_size = args.batch_size
    cube_size = (width, width, thickness)
    start_points = (sp_x, sp_y, sp_z)
    end_points = (ep_x, ep_y, ep_z)

    offset_w = width // 2
    offset_t = thickness // 2

    files = sorted(os.listdir(root))
    batch_num = n_cubes // batch_size

    start = time.time()
    num = 0
    with tqdm(total=n_cubes, desc=root.split('/')[-1]) as pbar:
        for i in range(1, batch_num + 1):
            # 随机生成cube的z切面起始坐标
            z = random.randint(start_points[2] + offset_t, end_points[2] - offset_t)
            imgs = np.array([np.array(Image.open(os.path.join(root, files[slice]))).astype(np.int32)
                             for slice in range(z - offset_t, z + offset_t)])
            # imgs = imgs.transpose(2, 1, 0)
            print("Load z slices of batch_" + str(i))

            for j in range(1, batch_size + 1):
                # 随机生成cube的x-y切面起始坐标
                x = random.randint(start_points[0] + offset_w, end_points[0] - offset_w)
                y = random.randint(start_points[1] + offset_w, end_points[1] - offset_w)
                # pdb.set_trace()
                print([x, y, z])
                tifffile.imwrite(os.path.join(target, 'volume-' + str(indx_start + num) + '.tiff'),
                                 imgs[:, y - offset_w: y + offset_w, x - offset_w: x + offset_w].astype(np.uint16))
                print("Volume_" + str(indx_start + num))
                num += 1
                pbar.update()
    end = time.time()
    print("Using time:{}".format(end - start))
    print('Final Done.')
