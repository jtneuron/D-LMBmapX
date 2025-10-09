import numpy as np
import pytorch_msssim
import torch


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, dim): return torch.diff(
        array, dim=dim)[:, :(H - 1), :(W - 1), :(D - 1)]

    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0] * (dy[1] * dz[2] - dz[1] * dy[2]) - dy[0] * (dx[1] * dz[2] -
                                                             dz[1] * dx[2]) + dz[0] * (dx[1] * dy[2] - dy[1] * dx[2])

    return det


def folds_count_metric(dvf):
    """
    dvf: B, 3, D, H, W
    """
    B = dvf.shape[0]
    count = 0
    for i in range(B):
        det = jacobian_determinant(dvf[i])
        count = count + (det <= 0).sum()
    return count / B


def folds_percent_metric(dvf):
    folds_num = folds_count_metric(dvf)
    return (folds_num / np.prod(dvf.shape[2:]).item()) * 100.


def SDLogJ_metric(dvf):
    B = dvf.shape[0]
    result = 0.
    for i in range(B):
        det = jacobian_determinant(dvf[i])
        det = torch.clamp(det, min=1e-9, max=1e9)
        det = torch.log(det)
        result = result + torch.std(det)
    return result / B


def mse_metric(predict, target):
    return torch.mean((predict - target) ** 2)


def ssim_metric(predict, target):
    return pytorch_msssim.ssim(predict, target, data_range=1., size_average=True)


def NMAE_meric(predict, target):
    return torch.mean(torch.abs(predict - target))


def PSNR_metric(predict, target, data_range):
    err = torch.mean((predict - target) ** 2)
    return 10 * torch.log10((data_range ** 2) / err)


if __name__ == '__main__':
    a = torch.randn(size=(1, 1, 128, 128, 128))
    b = torch.randn(size=(1, 1, 128, 128, 128))
    print(ssim_metric(a, b))
