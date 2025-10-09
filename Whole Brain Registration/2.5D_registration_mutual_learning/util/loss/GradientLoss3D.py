import torch
from torch import nn


class GradientLoss3D(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        super(GradientLoss3D, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, displacement_vector_field):
        dy = torch.abs(displacement_vector_field[:, :, 1:, :, :] - displacement_vector_field[:, :, :-1, :, :])
        dx = torch.abs(displacement_vector_field[:, :, :, 1:, :] - displacement_vector_field[:, :, :, :-1, :])
        dz = torch.abs(displacement_vector_field[:, :, :, :, 1:] - displacement_vector_field[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad = grad * self.loss_mult
        return grad
