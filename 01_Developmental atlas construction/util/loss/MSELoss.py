import torch
from torch import nn


class MSELoss(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_true - y_pred) ** 2)
