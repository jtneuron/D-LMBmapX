import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, background=False, weight=None, smooth=1e-4, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.background = background
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: B, C, D, H, W
        target   B, C, D, H, W
        """

        assert predict.shape == target.shape, "predict and target must have the same shape"

        num_classes = target.shape[1]

        total_dice = 0.

        weight = torch.tensor([1.] * num_classes) if self.weight is None else self.weight

        start = 0 if self.background else 1

        for i in range(start, num_classes):
            _dice = self.dice_coefficient(predict[:, i], target[:, i])
            total_dice = total_dice + weight[i] * _dice

        mean_dice = total_dice / num_classes if self.background else total_dice / (num_classes - 1)

        if self.reduction == 'mean':
            mean_dice = mean_dice.mean()

        return 1. - mean_dice

    def dice_coefficient(self, predict, target):
        """
        predict: B, D, H, W
        target: B, D, H, W
        """
        B = predict.shape[0]

        # B, D, H, W -> B, D*H*W
        predict = predict.view(B, -1)

        # B, D, H, W -> B, D*H*W
        target = target.view(B, -1)

        intersection = (predict * target).sum(dim=1)

        return (2. * intersection + self.smooth) / (predict.sum(dim=1) + target.sum(dim=1) + self.smooth)
