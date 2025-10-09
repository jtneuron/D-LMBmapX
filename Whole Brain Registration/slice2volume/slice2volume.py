# 先切片找landmark，再用ransac找最合适的切面，再ants在附近寻找的正交的切面，再在附近寻找最合适的切面
import subprocess
import os
import random
import shutil
from argparse import ArgumentParser
from collections import defaultdict
import ants
import cv2
import itk
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from skimage.feature import corner_peaks
from tqdm import tqdm
import copy
from utils.helper import conv, harris_corners, harris_corners_torch
from utils.utils import (
    cal_MI,
    clear_dir,
    generate_nearby_normals,
    get_slice,
    img_norm,
    sitk_read,
    sitk_write,
)
import json

PI = 3.1415927
USE_TRANSLATE = False

parser = ArgumentParser()
parser.add_argument('--moving_image_root_path',
                    default='',
                    help='moving_image_root_path')
parser.add_argument('--moving_image_trans_root_path',
                    default='',
                    help='moving_image_trans_root_path')
parser.add_argument('--fixed_image_path',
                    default='',
                    help='fixed_image_path')
parser.add_argument('--ave_anno_path',
                    default='',
                    help='ave_anno_path')
parser.add_argument('--soma_path',
                    default="",
                    help='soma_path')
parser.add_argument('--output_path',
                    default='',
                    help="output_path")
opt = parser.parse_args()

class Plane:
    """
    Implementation of planar RANSAC.

    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000, shuffle=True):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []
        
        if shuffle:
            pts = np.random.permutation(pts)

        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.inliers


def ants_reg_with_trans(fixed, moving, params, interpolate="nearestNeighbor"):
    fixed = ants.from_numpy(fixed)
    moving = ants.from_numpy(moving)
    return ants.apply_transforms(
        transformlist=params,
        fixed=fixed,
        moving=moving,
        interpolator=interpolate
    ).numpy()
    
def process(img, shape):
    assert len(img.shape) == 3
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray = F.interpolate(torch.from_numpy(gray[None, None, ...]), shape).numpy()[0, 0]
    gray = 255 - gray
    return gray


def harris_sift(img1, sift=None):
    if sift is None:
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.1, edgeThreshold=10, sigma=1.6)
    kp1 = corner_peaks(
        harris_corners_torch(img1), threshold_rel=0.01, exclude_border=16, min_distance=2
    )
    cvkp1 = [cv2.KeyPoint(x=float(p[1]), y=float(p[0]), size=20) for p in kp1]
    kp, desc = sift.compute(img1, cvkp1)
    return kp, desc


def cal_metric(matches, f, m):
    half = matches[: len(matches) // 2]
    res = sum([x.distance for x in half]) / len(half)

    f = np.ravel(f)
    m = np.ravel(m)
    mi = metrics.mutual_info_score(f, m)
    mi_d = 1.0 / mi
    return res / len(matches) * mi_d


def is_inside(x_range, y_range, pos):
    x1, x2 = x_range
    y1, y2 = y_range
    return pos[0] >= x1 and pos[0] <= x2 and pos[1] >= y1 and pos[1] <= y2


# clip match
def clip_matches(matches, kp_moving, kp_fixed, x_range, y_range):
    # x_range 横
    kp_moving = [p.pt[:] for p in kp_moving]
    kp_fixed = [p.pt[:] for p in kp_fixed]
    return filter(
        lambda x: 
        is_inside(x_range, y_range, kp_moving[x.queryIdx]) and is_inside(x_range, y_range, kp_fixed[x.trainIdx]),
        matches
    )

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



# 病理图像做moving
moving_image_root_path = opt.moving_image_root_path
# 病理迁移图像
moving_image_trans_root_path = opt.moving_image_trans_root_path
# 平均脑切片做fixed
fixed_image_path = opt.fixed_image_path
# 大脑边缘
ave_anno_path = opt.ave_anno_path
output_path = opt.output_path
soma_path = opt.soma_path

moving_image_paths = [
    os.path.join(moving_image_root_path, path) for path in os.listdir(moving_image_root_path)
]

if moving_image_trans_root_path:
    USE_TRANSLATE = True
    moving_image_paths = [
        os.path.join(moving_image_trans_root_path, path) 
        for path in os.listdir(moving_image_trans_root_path)
    ]


# 设置微调的角度
base_normal = [0, 0, 1]
angle_range_degrees = 5
step_degrees = 0.5
nearby_normals, angles = generate_nearby_normals(
    base_normal, angle_range_degrees, step_degrees
)

result_dict_part1 = defaultdict(dict)

fixed_image_3d = sitk_read(fixed_image_path).transpose(0, 2, 1)[::-1, ::-1, :]
ave_anno = sitk_read(ave_anno_path).transpose(0, 2, 1)[::-1, ::-1, :].astype(np.uint32)
ave_anno = np.ascontiguousarray(ave_anno)
# test
if fixed_image_3d.max() <= 1:
    fixed_image_3d *= 255.

a, b, c = fixed_image_3d.shape  # 320 528 456


MIN_KEY_POINT = 5
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
try:
    with open(os.path.join(output_path, "after_landmark", "result_dict_part1.json"), "r") as f:
        result = json.load(f)
        result = defaultdict(dict, result)
except Exception as e:
    print(e)
    result = defaultdict(dict)
for moving_path in tqdm(moving_image_paths):

    if USE_TRANSLATE:
        img_name = os.path.basename(moving_path).split('.')[0].split('_translated')[0]
    else:
        img_name = os.path.basename(moving_path).split('.')[0]
        
    if img_name in result.keys():
        continue

    os.makedirs(os.path.join(output_path,"after_landmark", img_name), exist_ok=True)
    clear_dir(os.path.join(output_path,"after_landmark", img_name))
    # shutil.copy(moving_path, os.path.join(output_path,"after_landmark", img_name, os.path.basename(moving_path)))

    img_moving = sitk.GetArrayFromImage(sitk.ReadImage(moving_path))

    # 迁移图像有些是3通道的
    if len(img_moving.shape) == 3:
        img_moving = img_moving[..., 0]
    
    img_moving = F.interpolate(torch.from_numpy(img_moving[None, None, ...]), (a, b)).numpy()[0, 0]
    
    # # TEST 归一化 
    # if not USE_TRANSLATE:
    #     img_moving = img_norm(img_moving)
    #     img_moving = 255 - img_moving
    #     img_moving = img_moving.astype(np.uint8)
    # else:
    #     img_moving = process(img_moving, shape=(a, b))
    #     img_moving = img_moving.astype(np.uint8)
    
    sitk_write(img_moving, os.path.join(output_path, "after_landmark", img_name, f"{img_name}.png"))
    kp_moving, desc_moving = harris_sift(img_moving)
    img_kp = cv2.drawKeypoints(img_moving, kp_moving, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite(os.path.join(output_path, "after_landmark", img_name, "keypoint.png"), img_kp)
    
    points_dict=defaultdict(dict)
    result_match_points = []
    for k in tqdm(range(c // 2)):
        if k < 77:
            continue
        img_ave_slice = fixed_image_3d[:,:,k].astype(np.uint8)
        kp_ave_slice, desc_ave_slice = harris_sift(img_ave_slice)
        matches = matcher.match(desc_moving, desc_ave_slice)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) >= MIN_KEY_POINT:
            
            img_kp = cv2.drawKeypoints(img_ave_slice, kp_ave_slice, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.imwrite(os.path.join(output_path, "after_landmark", img_name, f"index_{k}_keypoint.png"), img_kp)
            
            
            img_matches = cv2.drawMatches(img_moving, kp_moving, img_ave_slice, kp_ave_slice, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(output_path, "after_landmark", img_name, f"index_{k}_matches.png"), img_matches)
            # plt.figure(figsize=(12, 6))
            # plt.imshow(img_matches)
            # plt.savefig(os.path.join(output_path,"after_landmark", img_name, f"index_{k}_match.png"))
            # plt.close()

            point_list = [p.trainIdx for p in matches]
            point_list = [[p.pt[1], p.pt[0], k] for p in np.array(kp_ave_slice)[point_list]]
            
            # 计算平均距离 (只计算比较好的匹配)
            matches = sorted(matches, key=lambda x: x.distance)
            
            avg_distance = sum([x.distance for x in matches[: len(matches) // 1]]) / len(matches)
            points_dict[avg_distance] = point_list
    
    
    # 对得到的切面landmark按平均距离排序，取前一半进行ransac
    sorted_distance = sorted(points_dict.keys())
    keys = sorted_distance[: len(sorted_distance) // 25]
    for key in keys:
        result_match_points += points_dict[key]

    result_match_points = np.array(result_match_points)
    plane = Plane()
    best_eq, _ = plane.fit(result_match_points)
    aa, bb, cc, dd = best_eq
    print(f"{aa}x + {bb}y + {cc}z + {dd} = 0")

    slice_index = (-dd - aa * (a / 2) - bb * (b / 2)) / cc
    slice_index = slice_index.astype(int).item()
    
    np.save(os.path.join(output_path, "after_landmark", img_name, "result_match_points.npy"), np.array(point_list))
    result[img_name]['index'] = slice_index
    sl = get_slice(fixed_image_3d, [aa, bb, cc], [a // 2, b // 2, slice_index], (b, a))[::-1, ::-1]
    sitk_write(sl, os.path.join(output_path, "after_landmark", img_name, f"slice{aa}_{bb}_{cc}_{dd}.png"))

    with open(os.path.join(output_path, 'after_landmark', 'result_dict_part1.json'), 'w') as f:
        json.dump(result, f, indent=4)

# test 
with open(os.path.join(output_path, 'after_landmark', 'result_dict_part1.json'), 'r') as f:
    result_dict_part1 = json.load(f)

p = os.path.join(output_path, 'after_landmark', 'result_dict_part1.json')
if os.path.exists(p):
    with open(p, 'r') as f:
        result_dict_part2 = json.load(f)
else:
    result_dict_part2 = defaultdict(dict)
# 做ants
for i, moving_image_path in enumerate(tqdm(moving_image_paths, desc="直接配准", position=0)):

    # name = os.path.basename(moving_image_path).split('.')[0]
    if USE_TRANSLATE:
        name = os.path.basename(moving_image_path).split('.')[0].split('_translated')[0]
    else:
        name = os.path.basename(moving_image_path).split('.')[0]

    img_path = os.path.join(output_path, "after_ants", name)
    if os.path.exists(img_path):
        continue
    # os.makedirs(img_path, exist_ok=True)
    # clear_dir(img_path)
    
    moving_image = sitk_read(moving_image_path).astype(np.float32)
    # 迁移图像有些是3通道的
    if len(moving_image.shape) == 3:
        moving_image = moving_image[..., 0]
    moving_image = F.interpolate(torch.from_numpy(moving_image[None, None, ...]), (a, b)).squeeze().numpy()

    # 中间增加一步操作，先扩大范围寻找正交的切面
    try:
        mid_index = result_dict_part1[name]['index']
        if mid_index < 0 or mid_index >= c // 2:
            raise Exception("mid_index out of range")
        
        from_index = mid_index - 0
        to_index = mid_index + 1
        from_index = 0 if from_index < 0 else from_index
        to_index = c // 2 if to_index > c // 2 else to_index
        
    except Exception as e:
        print(e)
        
        from_index = 0
        to_index = c // 2

    best_metric = None
    for k in tqdm(range(from_index, to_index)):

        img_slice_origin = fixed_image_3d[:, :, k].astype(np.float32)
        img_slice = img_norm(img_slice_origin)
        

        mytx = ants.registration(fixed=ants.from_numpy(img_slice),
                                moving=ants.from_numpy(moving_image),
                                type_of_transform="Affine",
                                # mask=ants.from_numpy(mask)
                                )
        registered_img = mytx['warpedmovout']

        mytx = ants.registration(fixed=ants.from_numpy(img_slice),
                                moving=registered_img,
                                type_of_transform="SyN",
                                # mask=ants.from_numpy(mask)
                                )
        registered_img = mytx['warpedmovout'].numpy()

        metric = -metrics.mutual_info_score(
            img_slice.astype(np.uint8).flatten(),
            registered_img.astype(np.uint8).flatten(),
        )

        # if 'metric' not in result_dict_part2[name] or metric < result_dict_part2[name]['metric']:
        if best_metric is None or metric < best_metric:
            best_metric = metric
            mid_index = k
        


    from_index = mid_index - 0
    to_index = mid_index + 1
    from_index = 0 if from_index < 0 else from_index
    to_index = c // 2 if to_index > c // 2 else to_index

    # img_chosen_origin = None
    img_chosen = None
    warp_img = None
    result_params = None
    result_anno = None

    for k in tqdm(range(from_index, to_index)):
            
        for j, normal in enumerate(tqdm(nearby_normals)):
            # center
            point = [a // 2, b // 2, k]
            img_slice_origin = get_slice(fixed_image_3d, normal, point, (b, a))[::-1, :]
            img_slice = img_norm(img_slice_origin)
            
            # mask = get_slice(ave_anno, normal, point, (b, a))[::-1, :]
            # mask[mask > 0] = 1
            # mask = mask.astype(np.uint8)

            params = []
            mytx = ants.registration(fixed=ants.from_numpy(img_slice),
                                    moving=ants.from_numpy(moving_image),
                                    type_of_transform="Affine",
                                    # mask=ants.from_numpy(mask)
                                    )
            params.append(mytx['fwdtransforms'])
            registered_img = mytx['warpedmovout']

            mytx = ants.registration(fixed=ants.from_numpy(img_slice),
                                    moving=registered_img,
                                    type_of_transform="SyN",
                                    # mask=ants.from_numpy(mask)
                                    )
            params.append(mytx['fwdtransforms'])
            registered_img = mytx['warpedmovout'].numpy()

            metric = -metrics.mutual_info_score(
                img_slice.astype(np.uint8).flatten(),
                registered_img.astype(np.uint8).flatten(),
            )

            if 'metric' not in result_dict_part2[name] or metric < result_dict_part2[name]['metric']:
                result_dict_part2[name]['index'] = k
                result_dict_part2[name]['metric'] = metric
                result_dict_part2[name]['normal'] = list(normal)
                result_dict_part2[name]['angles'] = list(np.array(angles[j]) * 180 / PI)
                
                img_chosen = img_slice_origin
                result_params = params
                result_anno = get_slice(ave_anno, normal, point, (b, a))[::-1, :]
                warp_img = registered_img

    os.makedirs(img_path, exist_ok=True)
    sitk_write(moving_image.astype(np.uint8), os.path.join(img_path, os.path.basename(moving_image_path)))

    # sitk_write(img_chosen_origin.astype(np.uint8), os.path.join(img_path, f"{int(result_dict_part2[name]['index'])}_origin.png"))

    if img_chosen is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(img_chosen[:, :].astype(np.uint8)),
            os.path.join(img_path, f"{int(result_dict_part2[name]['index'])}.png"),
        )
    if warp_img is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(warp_img[:, :].astype(np.uint8)),
            os.path.join(img_path, "warp_img.png"),
        )
    if result_params is not None:
        p = os.path.join(soma_path, f"{name}.jpg")
        if not os.path.exists(p):
            p = os.path.join(soma_path, f"{name}.png")
        if not os.path.exists(p):
            p = os.path.join(soma_path, f"{name}.tiff")
        if not os.path.exists(p):
            raise Exception(f"Soma image not found for {name} at {p}")
        
        soma = sitk.GetArrayFromImage(sitk.ReadImage(p))
        soma = F.interpolate(torch.from_numpy(soma[None, None, :]), (a, b)).numpy()[0, 0]
        # warp_soma = itk.transformix_filter(soma, result_params)
        warp_soma = ants_reg_with_trans(img_slice, soma, result_params[0])
        warp_soma = ants_reg_with_trans(img_slice, warp_soma, result_params[1])

        sitk.WriteImage(
            sitk.GetImageFromArray(soma), os.path.join(img_path, "soma.png")
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(warp_soma.astype(np.uint8)), os.path.join(img_path, "warp_soma.png")
        )
    if result_anno is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(result_anno.astype(np.uint16)), os.path.join(img_path, "anno.png")
        )
    print(os.path.basename(moving_image_path))
    print(result_dict_part2[name]['index'])
    print("\n")
    # np.save(os.path.join(output_path, "after_elastix", "result.npy"), result)
    with open(os.path.join(output_path, "after_ants", "result_dict_part2.json"), 'w') as f:
        json.dump(result_dict_part2, f, indent=4)
        
    
    cmd = "rm -f /tmp/*.nii.gz"
    subprocess.run(cmd, shell=True, check=False)
    cmd = "rm -f /tmp/*.mat"
    subprocess.run(cmd, shell=True, check=False)


