import numpy as np
import torch


def jacobian_determinant_2d(vf):
    _, H, W = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, dim): return torch.diff(
        array, dim=dim)[:, :(H - 1), :(W - 1)]

    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1

    # Compute determinant at each spatial location
    det = dx[0] * dy[1] - dx[1] * dy[0]

    return det


def folds_count_metric_2d(dvf):
    B = dvf.shape[0]
    count = 0
    for i in range(B):
        det = jacobian_determinant_2d(dvf[i])
        count = count + (det <= 0).sum()
    return count / B


def folds_percent_metric_2d(dvf):
    folds_num = folds_count_metric_2d(dvf)
    return (folds_num / np.prod(dvf.shape[2:]).item()) * 100.


def SDLogJ_metric_2d(dvf):
    B = dvf.shape[0]
    result = 0.
    for i in range(B):
        det = jacobian_determinant_2d(dvf[i])
        det = torch.clamp(det, min=1e-9, max=1e9)
        det = torch.log(det)
        result = result + torch.std(det)
    return result / B


if __name__ == '__main__':
    a = torch.randn(size=(1, 1, 128, 128, 128))
    b = torch.randn(size=(1, 1, 128, 128, 128))
