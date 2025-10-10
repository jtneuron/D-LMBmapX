import os
import cv2
import cc3d
import torch
import tifffile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.morphology import skeletonize, thin


def thined_component(stack, vol):
    labels = cc3d.connected_components(stack)
    print(labels.shape)
    print('\n')
    shapes = labels.shape
    labels = labels.flatten()
    print(labels.shape)
    print('\n')
    mark = [0] * (labels.max() + 1)
    print(len(mark))
    print('\n')
    for i in labels:
        mark[i] += 1
    for k in range(len(labels)):
        if mark[labels[k]] < vol:
            labels[k] = 0
    labels[labels > 0] = 255

    return labels.reshape(shapes)


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


def gaussian(x, kernal, flg_min):
    x = cv2.blur(x, (kernal, kernal))
    mi = x.min()
    mx = x.max()
    imin = flg_min * mx + (1 - flg_min) * mi

    return x

    # .cpu()[0][0]).astype(np.uint8)


def gaussian_slice(x, kernel_size):
    for i in range(x.shape[0]):
        x[:, :, i] = cv2.blur(x[:, :, i], (kernel_size, kernel_size))
    return x


# thinned_partial = thin(labels, max_iter=2)
parser = argparse.ArgumentParser(description='Skeletonization',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--source', type=str, default=None, dest='source',
                    help='source directory of the brain slices')
parser.add_argument('-t', '--target', type=str, default=None, dest='target',
                    help='target directory of the cropped volumes')

parser.add_argument('--gpu', type=int, default=0, help='要使用的 GPU ID (默认值: 0)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--overlap', type=int, default=8, help='overlap')


args = parser.parse_args()
base = args.source
target = args.target

print(base)
print(target)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
labels = read_tiff_stack(base)
print("successfully read in!")

batch_size = args.batch_size
overlap = args.overlap
step = batch_size - overlap
depth, _, _ = labels.shape
processed_labels = np.zeros_like(labels, dtype=np.uint8)
print(processed_labels.shape)


for start in tqdm(range(0, depth, step), desc=target):
    print(start)
    end = min(start + batch_size, depth)
    batch_labels = labels[start:end, :, :]
    batch_labels = gaussian_slice(batch_labels, 5)
    skeleton = skeletonize(batch_labels)
    x = torch.Tensor(skeleton.reshape(1, 1, *skeleton.shape)).to(device)
    p1 = torch.nn.functional.max_pool3d(x, (3, 1, 1), 1, (1, 0, 0))
    p2 = torch.nn.functional.max_pool3d(x, (1, 3, 1), 1, (0, 1, 0))
    p3 = torch.nn.functional.max_pool3d(x, (1, 1, 3), 1, (0, 0, 1))
    min_pool_x = torch.min(torch.min(p1, p2), p3)
    x = torch.max(torch.max(p1, p2), p3)
    # x = gaussian(np.array(x.cpu()[0][0]).astype(np.uint8), 3, 0.5)
    # x = (x - x.min()) / (x.max() - x.min()) * 255
    x = np.array(x.cpu()[0][0]).astype(np.uint8)
    x[x > 0] = 255
    x[x <= 0] = 0
    x = thined_component(x.astype(np.uint8), 50).astype(np.uint8)
    processed_labels[start:end, :, :] = np.maximum(processed_labels[start:end, :, :], x)

tifffile.imwrite(target, processed_labels)

print('Done.')
