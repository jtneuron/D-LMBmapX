import os
from argparse import ArgumentParser

import ants
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json
import toml

threshold = 0.45


def sitk_read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def sitk_write(arr, path):
    sitk.WriteImage(sitk.GetImageFromArray(arr), path)

def reg_pipeline(fixed, moving, type_of_transform, *to_transforms, 
                 **registration_kwargs):
    """
    fixed: 固定图像
    moving: 浮动图像
    type_of_transform: 变换类型
    mask: mask
    to_transforms: 需要施加形变的图像
    """
    mytx = ants.registration(fixed=fixed,
                             moving=moving,
                             type_of_transform=type_of_transform,
                             **registration_kwargs)  

    registered_img = mytx["warpedmovout"]

    result = [ants.apply_transforms(fixed=to_transform,
                                    moving=to_transform,
                                    transformlist=mytx["fwdtransforms"],
                                    interpolator="nearestNeighbor") 
              for to_transform in to_transforms]
    
    for i in mytx["fwdtransforms"]:
        if os.path.exists(i):
            os.remove(i)
    for i in mytx["invtransforms"]:
        if os.path.exists(i):
            os.remove(i)

    return registered_img, result

def ave_all_brain(*data):
    if isinstance(data[0], str):
        data = sitk_read(data)
    
    dtype = data[0].dtype
    base = data[0].astype(np.float64)
    count = len(data)
    for x in data[1:]:
        base += x
    return (base / count).astype(dtype=dtype)
    
    



def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir',
                        help='dataset config')
    parser.add_argument('--output_dir',
                        help='config path to images')
    parser.add_argument('--mode',
                    default='SyN',
                    choices=["Affine", "SyN"],
                    help="mode of registration")
    args = parser.parse_args()


    # -----读取config-------
    if args.data_dir.endswith('toml'):
        with open(args.data_dir, 'r') as f:
            dataset_config = toml.load(f)
    elif args.data_dir.endswith('json'):
        with open(args.data_dir, 'r') as f:
            dataset_config = json.load(f)
    else:
        raise Exception("dataset config file format error")
        
    data_paths = dataset_config["moving"]
    atlas_path = dataset_config.get("atlas", None)
    region_number = dataset_config["region_number"]

    if atlas_path:
        if not os.path.exists(atlas_path):
            raise ValueError(f"no {atlas_path}")
        atlas = sitk_read(atlas_path)
    else:
        # 无atlas，默认全部平均作为ave
        atlas = ave_all_brain(*data_paths)



    imgshape = atlas.shape
    datasets_size = len(data_paths)
    ave = np.zeros(shape=imgshape)
    ave_multi_region = {str(i): np.zeros(imgshape, 'float') for i in range(1, region_number + 1)}
    ave_label = np.zeros(imgshape, dtype=np.uint8)



    print(f"{datasets_size = }")
    print(f"use {args.mode}")

    for moving_path in tqdm(data_paths):
        name = os.path.basename(moving_path).split(".")[0]
        moving = ants.from_numpy(sitk_read(moving_path))
        fixed = ants.from_numpy(atlas)
        moving_mask = ants.from_numpy(sitk_read(os.path.join(moving_path.replace(".nii.gz", "_label.nii.gz"))))
        
        # mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform=args.mode)
        # warped_mask = ants.apply_transforms(fixed=moving_mask, moving=moving_mask,
        #                                     transformlist=mytx['fwdtransforms'], interpolator='nearestNeighbor').numpy()

        warped_moving, (warped_mask, ) = reg_pipeline(fixed, moving, args.mode, moving_mask)
        warped_moving = warped_moving.numpy()
        warped_mask = warped_mask.numpy()
        ave += warped_moving

        for i in range(1, region_number + 1):
            ave_multi_region[str(i)] += (warped_mask == i) * i

        mid_path = f"{args.mode}_result"
        save_dir = os.path.join(args.output_dir, mid_path, name)
        os.makedirs(save_dir, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(warped_moving.astype(np.uint8)), os.path.join(save_dir, name+".nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(warped_mask.astype(np.uint8)), os.path.join(save_dir, name+"_label.nii.gz"))


    ave /= datasets_size

    for i in range(1, region_number + 1):
        value = ave_multi_region[str(i)]
        value /= datasets_size

        value[value < threshold * i] = 0
        value[value != 0] = np.max(value)

        # 形态学后处理
        import skimage
        value = skimage.morphology.closing(value)
        value = skimage.morphology.opening(value)
        ave_label[value != 0] = i


    mid_path = f"{args.mode}_ave"

    save_dir = os.path.join(args.output_dir, mid_path)
    os.makedirs(save_dir, exist_ok=True)

    sitk_write(ave.astype(np.uint8), os.path.join(save_dir, args.mode+'_ave.nii.gz'))
    sitk_write(ave_label.astype(np.uint8), os.path.join(save_dir, args.mode+'_ave_label.nii.gz'))

if __name__ == "__main__":
    main()