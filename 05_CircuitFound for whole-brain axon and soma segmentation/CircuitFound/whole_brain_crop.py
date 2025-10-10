import os
import argparse
import numpy as np
from tifffile import imwrite
from tqdm import tqdm
from PIL import Image


def crop_3d_brain_fixed(input_dir, output_root, x_start, y_start, width, height):
    """
    对3D脑图像进行固定尺寸的裁剪

    参数:
        input_dir: 输入文件夹路径(包含TIFF切片的文件夹)
        output_dir: 输出文件夹路径
        x_start, y_start: 裁剪区域左上角坐标
        width, height: 裁剪区域的宽度和高度
    """
    # 计算结束点坐标
    x_end = x_start + width
    y_end = y_start + height
    threshold_value = 125

    dir_name = input_dir.split('/')[-1]
    output_dir = output_root + '/' + dir_name + '_crop_final'

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    elif os.path.exists(output_dir) and len(os.listdir(output_dir)) == len(os.listdir(input_dir)):
        print('!!!!!!!!!!!!!!!!!!! {} has existed!'.format(dir_name))
        exit()

    # 获取所有TIFF文件并按名称排序
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])

    print(f"开始处理文件夹: {input_dir}")
    print(f"将裁剪区域: X={x_start}-{x_end}, Y={y_start}-{y_end} (宽{width}, 高{height})")
    print(f"找到{len(tiff_files)}个TIFF文件")

    if '1300' in input_dir:
        print("------- process axon -----------")
        for filename in tqdm(tiff_files, desc= dir_name):
            # 读取TIFF文件
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img = np.array(img)

            img = (img >= threshold_value) * 255
            img = img.astype(np.uint8)

            # 检查图像尺寸是否足够大
            if img.shape[0] < y_end or img.shape[1] < x_end:
                print(f"警告: {filename} 尺寸{img.shape}小于裁剪区域，跳过")
                continue

            # 执行裁剪
            cropped_img = img[y_start:y_end, x_start:x_end]

            # 保存裁剪后的图像
            output_path = os.path.join(output_dir, filename)
            imwrite(output_path, cropped_img)
    else:
        for filename in tqdm(tiff_files, desc=dir_name):
            # 读取TIFF文件
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img = np.array(img)

            # 检查图像尺寸是否足够大
            if img.shape[0] < y_end or img.shape[1] < x_end:
                print(f"警告: {filename} 尺寸{img.shape}小于裁剪区域，跳过")
                continue

            # 执行裁剪
            cropped_img = img[y_start:y_end, x_start:x_end]

            # 保存裁剪后的图像
            output_path = os.path.join(output_dir, filename)
            imwrite(output_path, cropped_img)

    print(f"成功处理: {filename} (原始尺寸: {img.shape}, 裁剪后: {cropped_img.shape})")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='3D脑图像固定尺寸裁剪工具')
    parser.add_argument('-i', '--input', required=True, help='输入文件夹路径(包含TIFF切片的3D脑文件夹)')
    parser.add_argument('-o', '--output', required=True, help='输出文件夹路径')
    parser.add_argument('--x', type=int, required=True, help='裁剪区域左上角X坐标')
    parser.add_argument('--y', type=int, required=True, help='裁剪区域左上角Y坐标')
    parser.add_argument('--width', type=int, required=True, help='裁剪区域宽度')
    parser.add_argument('--height', type=int, required=True, help='裁剪区域高度')

    args = parser.parse_args()

    # 调用裁剪函数
    crop_3d_brain_fixed(
        input_dir=args.input,
        output_root=args.output,
        x_start=args.x,
        y_start=args.y,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()