import numpy as np
from tifffile import imwrite
import time
from scipy import ndimage
from PIL import Image
import os
import multiprocessing as mp
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import reconstruction, dilation, erosion, cube, square, ball, disk
import argparse
Image.MAX_IMAGE_PIXELS = None


##########开操作##########
def remove_axon(img, size):
    img_ero = binary_erosion(img, cube(size))
    img_dil = binary_dilation(img_ero, cube(size))
    return img_dil.astype(np.uint8)


##########闭操作##########
def closeing_filled(img):
    img_dil = binary_dilation(img, ball(2))
    img_ero = binary_erosion(img_dil, ball(2))
    return img_ero.astype(np.uint8)


##########孔洞填充##########
def fill_holes(data):
    return ndimage.binary_fill_holes(data)


##########2d孔洞填充##########
def fill_holes_2d(image_1):
    # 二值化
    binary_image = (image_1 > 0).astype(np.uint8)

    # fmost
    # structure_element = square(6)

    # lsfm
    structure_element = disk(10)

    # Apply binary dilation to the inner part of the array
    dilated_result = binary_dilation(binary_image, structure=structure_element)

    # 应用连通组件分析来标记连通区域
    labeled_volume, num_features = ndimage.label(dilated_result)
    # 获取每个标签区域的大小
    sizes = sum(dilated_result, labeled_volume, range(num_features + 1))

    # 过滤掉大于阈值的连通区域
    mask_threshold = 255
    mask_sizes = sizes >= mask_threshold  # 假定连通区域大255个像素
    mask_sizes[0] = 0  # 背景标签不做处理

    # 创建新的mask，其中只包含过滤后的连通区域
    dilated_result = mask_sizes[labeled_volume]

    # 腐蚀操作
    dilated_result = binary_erosion(dilated_result, structure=structure_element)

    # Update the dilated array
    binary_image = dilated_result

    return binary_image.astype(np.uint8)


##########去除小型/大型或线状的过分割噪声/axon#########
def remove_noise(img, flag=None, thres=None):
    # 应用连通组件分析来标记连通区域
    labeled_volume, num_features = ndimage.label(img)
    # 获取每个标签区域的大小
    sizes = ndimage.sum(img, labeled_volume, range(num_features + 1))

    mask_sizes1 = None
    mask_sizes2 = None

    if flag == 'low':
        # 过滤掉小于阈值的连通区域
        mask_low_threshold = thres
        mask_sizes1 = sizes > mask_low_threshold
    elif flag == 'high':
        # 过滤掉大于阈值的连通区域   
        mask_high_threshold = thres
        mask_sizes2 = sizes <= mask_high_threshold

    if mask_sizes1 is not None and mask_sizes2 is not None:
        mask_sizes = np.logical_and(mask_sizes1, mask_sizes2)
    elif mask_sizes1 is not None:
        mask_sizes = mask_sizes1
    else:
        mask_sizes = mask_sizes2

    mask_sizes[0] = 0  # 背景标签不做处理

    # 创建新的3D mask，其中只包含过滤后的连通区域
    result = mask_sizes[labeled_volume] > 0

    return result.astype(np.uint8)


def read_tiff_files(root):
    res = []
    for f in sorted(os.listdir(root)):
        res.append(np.array(Image.open(path(root, f))))
    return np.array(res)


def path(*args):
    return os.path.join(*args)


def sliding_window_3d(mask, window_size=(64, 64, 64), stride=(32, 32, 32), post_process_code=None):
    z_size, y_size, x_size = mask.shape
    z_win, y_win, x_win = window_size
    z_stride, y_stride, x_stride = stride

    # 计算每个维度上滑动窗口操作的次数
    z_steps = (z_size - z_win) // z_stride + 1
    y_steps = (y_size - y_win) // y_stride + 1
    x_steps = (x_size - x_win) // x_stride + 1

    print(z_steps, y_steps, x_steps)

    total_operations = z_steps * y_steps * x_steps

    print(f"Total operations needed: {total_operations}")

    processed_mask = np.zeros_like(mask, dtype=np.uint8)

    operation_count = 0

    # define the start of each window
    z_list = [z for z in range(0, z_size - z_win + 1, z_stride)] + [z_size - z_win]
    y_list = [y for y in range(0, y_size - y_win + 1, y_stride)] + [y_size - y_win]
    x_list = [x for x in range(0, x_size - x_win + 1, x_stride)] + [x_size - x_win]

    for z in z_list:
        for y in y_list:
            for x in x_list:
                operation_count += 1
                start_time = time.time()

                window = mask[z:z + z_win, y:y + y_win, x:x + x_win]

                # Apply post-processing function
                if post_process_code:
                    window = post_process_code(window)

                # Merge the processed window using logical "or"
                processed_mask[z:z + z_win, y:y + y_win, x:x + x_win] = np.logical_or(
                    processed_mask[z:z + z_win, y:y + y_win, x:x + x_win], window)

                end_time = time.time()

                print(
                    f"Operation {operation_count}/{total_operations} completed in {end_time - start_time:.4f} seconds.")

    return processed_mask.astype(mask.dtype)


# Post-processing function
def post_process_code(window):
    window = remove_axon(window, 4)
    return window

def save_slice(slice_data, output_folder, slice_index):
    # 将切片数据保存为uint8类型的TIFF文件
    slice_path = os.path.join(output_folder, f'{slice_index:04d}.tiff')
    imwrite(slice_path, slice_data.astype(np.uint8))


parser = argparse.ArgumentParser()
parser.add_argument('--imgs_path', type=str, default=None, help='seg soma dir path')
parser.add_argument('--output_path', type=str, default=None, help='output dir path')
args = parser.parse_args()


imgs_path = args.imgs_path
output_path = args.output_path

os.makedirs(output_path, exist_ok=True)

img = read_tiff_files(imgs_path)
print("Successfully read in!")
print(img.shape)

img[img == 255] = 1
# 定义滑动窗口的大小和步长
window_size = (128, 128, 128)
overlap = 32  # 最大胞体大小
stride = [i - overlap for i in window_size]

processed_mask = sliding_window_3d(img, window_size, stride, post_process_code)

processed_mask[processed_mask > 0] = 255

tasks = [(processed_mask[i, :, :], output_path, i) for i in range(processed_mask.shape[0])]

# print("test:", tasks[100])

# 设置进程池大小（可以根据需要调整）
pool_size = 4  # 或者 None 使用系统默认的核心数

# 使用多进程保存切片
# 创建进程池
pool = mp.Pool(processes=8)

# 使用 starmap_async 方法进行并行处理
results = pool.starmap_async(save_slice, tasks)

# 获取处理结果
output = results.get()

# 关闭进程池
pool.close()
pool.join()