import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# class NCCLoss_1(nn.Module):
#     def __init__(self, spatial_dims=3, kernel_size=9):
#         super(NCCLoss_1, self).__init__()
#         self.loss_func = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=spatial_dims,
#                                                                           kernel_size=kernel_size)
#
#     def forward(self, y_pred, y_true):
#         return self.loss_func(y_pred, y_true)


class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, spatial_dims=3, kernel_size=9):
        super(NCCLoss, self).__init__()
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size

    def forward(self, y_pred, y_true):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        # win = [9] * ndims if self.win is None else self.win
        win = [self.kernel_size] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.data.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


# if __name__ == '__main__':
#     x = torch.randn((8, 1, 128, 128, 128)).cuda()
#     y = torch.randn((8, 1, 128, 128, 128)).cuda()
#     output1 = NCCLoss_1(spatial_dims=3, kernel_size=9)(x, y)
#     output2 = NCCLoss(spatial_dims=3, kernel_size=9)(x, y)
#     print(output1)
#     print(output2)
