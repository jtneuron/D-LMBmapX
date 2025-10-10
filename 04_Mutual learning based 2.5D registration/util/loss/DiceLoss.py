import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        B = predict.shape[0]
        dim = tuple([i for i in range(1, target.ndim)])
        numerator = torch.sum(predict * target, dim=dim) * 2
        denominator = torch.sum(predict, dim=dim) + torch.sum(target, dim=dim) + 1e-6
        return torch.sum(1 - numerator / denominator) / B
