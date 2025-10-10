import pdb

import numpy as np
import math
import time
from scipy.ndimage import convolve
from tqdm import tqdm
from torch.nn.functional import conv3d
import torch


def hog3d_GPU(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins, step_size=None):
    """
    Inputs

    vox_volume : a 	[x x y x z] tensor defining voxels with values in the range 0-1
    cell_size : size of a 3d cell (int)
    block_size : size of a 3d block defined in cells
    theta_histogram_bins : number of bins to break the angles in the xy plane - 180 degrees
    phi_histogram_bins : number of bins to break the angles in the xz plane - 360 degrees
    step_size : OPTIONAL integer defining the number of cells the blocks should overlap by.
	"""

    # 开始时间
    start_time = time.time()
    device = torch.device("cuda:2")

    # 将vox_volume转换为torch张量，并移动到GPU
    vox_volume = torch.tensor(vox_volume, device=device).float()

    if step_size is None:
        step_size = block_size

    c = cell_size
    b = block_size

    sx, sy, sz = vox_volume.shape

    num_x_cells = math.floor(sx / cell_size)
    num_y_cells = math.floor(sy / cell_size)
    num_z_cells = math.floor(sz / cell_size)

    # Get cell positions
    x_cell_positions = np.array(list(range(0, (num_x_cells * cell_size), cell_size)))
    y_cell_positions = np.array(list(range(0, (num_y_cells * cell_size), cell_size)))
    z_cell_positions = np.array(list(range(0, (num_z_cells * cell_size), cell_size)))

    # Get block positions
    x_block_positions = (x_cell_positions[0: num_x_cells: block_size])
    y_block_positions = (y_cell_positions[0: num_y_cells: block_size])
    z_block_positions = (z_cell_positions[0: num_z_cells: block_size])

    # Check if last block in each dimension has enough voxels to be a full block. If not, discard it.
    if x_block_positions[-1] > ((sx + 1) - (cell_size * block_size)):
        x_block_positions = x_block_positions[:-2]
    if y_block_positions[-1] > ((sy + 1) - (cell_size * block_size)):
        y_block_positions = y_block_positions[:-2]
    if z_block_positions[-1] > ((sz + 1) - (cell_size * block_size)):
        z_block_positions = z_block_positions[:-2]

    # Number of blocks
    num_x_blocks = len(x_block_positions)
    num_y_blocks = len(y_block_positions)
    num_z_blocks = len(z_block_positions)

    # Create 3D gradient vectors
    # X filter and vector
    x_filter = torch.zeros((1, 1, 3, 3, 3), device=vox_volume.device)
    x_filter[0, 0, 0, 1, 1], x_filter[0, 0, 2, 1, 1] = 1, -1
    x_vector = conv3d(vox_volume.unsqueeze(0).unsqueeze(0), x_filter, padding=1)

    # Y filter and vector
    y_filter = torch.zeros((1, 1, 3, 3, 3), device=vox_volume.device)
    y_filter[0, 0, 1, 0, 1], y_filter[0, 0, 1, 2, 1] = 1, -1
    y_vector = conv3d(vox_volume.unsqueeze(0).unsqueeze(0), y_filter, padding=1)

    # Z filter and vector
    z_filter = torch.zeros((1, 1, 3, 3, 3), device=vox_volume.device)
    z_filter[0, 0, 1, 1, 0], z_filter[0, 0, 1, 1, 2] = 1, -1
    z_vector = conv3d(vox_volume.unsqueeze(0).unsqueeze(0), z_filter, padding=1)

    magnitudes = torch.sqrt(x_vector ** 2 + y_vector ** 2 + z_vector ** 2)
    magnitudes = magnitudes.squeeze(0).squeeze(0)

    kernel_size = 3
    voxel_filter_value = 1 / (kernel_size * kernel_size * kernel_size)
    voxel_filter = torch.full((kernel_size, kernel_size, kernel_size), voxel_filter_value, device=vox_volume.device)
    voxel_filter = voxel_filter.unsqueeze(0).unsqueeze(0)  # 添加额外的维度以适配conv3d的期望输入形状
    # 使用卷积计算权重，注意卷积的输入需要是5维的，格式为(batch_size, channel, depth, height, width)
    weights = conv3d(vox_volume.unsqueeze(0).unsqueeze(0), voxel_filter, padding=1)
    # 移除前两个用于扩展的维度，还原成原来的形状
    weights = weights.squeeze(0).squeeze(0)
    # 添加1以计算最终权重
    weights += 1

    # 使用torch.stack来合并向量，形成梯度向量
    grad_vector = torch.stack((x_vector, y_vector, z_vector), dim=-1)

    # 计算theta，使用arccos并保证值在合适的范围内
    theta = torch.acos(grad_vector[..., 2].clamp(-1.0, 1.0))
    theta = theta.squeeze(0).squeeze(0)

    # 计算phi，使用atan2，并将结果调整到[0, 2*pi]的范围内
    phi = torch.atan2(grad_vector[..., 1], grad_vector[..., 0]) + torch.pi
    phi = phi.squeeze(0).squeeze(0)

    # Binning
    b_size_voxels = int(c * b)
    t_hist_bins = math.pi / theta_histogram_bins
    p_hist_bins = (2 * math.pi) / phi_histogram_bins

    block_inds = torch.zeros((num_x_blocks * num_y_blocks * num_z_blocks, 3), device=vox_volume.device)
    i = 0
    for z_block in range(num_z_blocks):
        for y_block in range(num_y_blocks):
            for x_block in range(num_x_blocks):
                block_inds[i] = torch.tensor(np.array(
                    [x_block_positions[x_block], y_block_positions[y_block], z_block_positions[z_block]]))
                i += 1

    num_blocks = len(block_inds)
    features = []

    # pdb.set_trace()

    for i in range(num_blocks):
        block_start_x = int(block_inds[i, 0])
        block_start_y = int(block_inds[i, 1])
        block_start_z = int(block_inds[i, 2])

        block_end_x = block_start_x + b_size_voxels
        block_end_y = block_start_y + b_size_voxels
        block_end_z = block_start_z + b_size_voxels

        full_empty = vox_volume[block_start_x:block_end_x, block_start_y:block_end_y, block_start_z:block_end_z]

        if torch.sum(full_empty) != 0 and torch.sum(full_empty) != full_empty.numel():
            feature = torch.zeros((b, b, b, theta_histogram_bins, phi_histogram_bins), device=vox_volume.device)
            t_weights = weights[block_start_x:block_end_x, block_start_y:block_end_y, block_start_z:block_end_z]
            t_magnitudes = magnitudes[block_start_x:block_end_x, block_start_y:block_end_y, block_start_z:block_end_z]
            t_theta = theta[block_start_x:block_end_x, block_start_y:block_end_y, block_start_z:block_end_z]
            t_phi = phi[block_start_x:block_end_x, block_start_y:block_end_y, block_start_z:block_end_z]

            # 创建网格
            l, m, n = torch.meshgrid(torch.arange(b_size_voxels, device=vox_volume.device),
                                     torch.arange(b_size_voxels, device=vox_volume.device),
                                     torch.arange(b_size_voxels, device=vox_volume.device), indexing='ij')

            cell_pos_x = torch.ceil(l / c).long() - 1
            cell_pos_y = torch.ceil(m / c).long() - 1
            cell_pos_z = torch.ceil(n / c).long() - 1

            hist_pos_theta = torch.ceil(t_theta[l, m, n] / t_hist_bins).long() - 1
            hist_pos_phi = torch.ceil(t_phi[l, m, n] / p_hist_bins).long() - 1

            # pdb.set_trace()

            # 校正直方图位置的范围
            hist_pos_theta = torch.clamp(hist_pos_theta, 0, theta_histogram_bins - 1)
            hist_pos_phi = torch.clamp(hist_pos_phi, 0, phi_histogram_bins - 1)

            # 更新特征张量
            # 使用布尔索引创建掩码，然后将满足条件的位置相加
            mask = (hist_pos_phi >= 0) & (hist_pos_phi < phi_histogram_bins) & (hist_pos_theta >= 0) & (
                        hist_pos_theta < theta_histogram_bins)
            feature[cell_pos_x[mask], cell_pos_y[mask], cell_pos_z[mask], hist_pos_theta[mask], hist_pos_phi[mask]] += \
            t_magnitudes[mask] * t_weights[mask]


            # 重塑 feature 张量以准备归一化
            feature = feature.view(-1, theta_histogram_bins, phi_histogram_bins)

            # 计算 L2 范数
            l2 = torch.norm(feature, p=2)

            # 归一化特征，如果 L2 范数不为零
            if l2 != 0:
                norm_feature = feature / l2
            else:
                norm_feature = feature

            # 重新将特征向量重塑为一维数组
            norm_feature = norm_feature.view(-1, theta_histogram_bins * phi_histogram_bins)

            features.append(norm_feature.detach().cpu().numpy())  # 将特征移回CPU并转换为NumPy数组，如果需要在GPU上保留，去掉`.cpu().numpy()`

    # 结束时间
    end_time = time.time()

    # 计算运行时间
    gpu_execution_time = end_time - start_time
    print("GPU executing time:", gpu_execution_time, " seconds")

    return features