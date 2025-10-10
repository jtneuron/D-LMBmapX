import torch


def dice_coefficient(predict, target):
    """
    predict: B, D, H, W
    target: B, D, H, W
    """
    B = predict.shape[0]

    # B, D, H, W -> B, D*H*W
    predict = predict.view(B, -1)

    # B, D, H, W -> B, D*H*W
    target = target.view(B, -1)

    intersection = (predict * target).sum()

    two_sum = predict.sum() + target.sum()

    return 2. * intersection / two_sum if two_sum > 1e-6 else torch.tensor(0., device=predict.device)

    # return (2. * intersection) / (predict.sum() + target.sum())


def dice_metric(predict, target, reduction="mean"):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """

    assert predict.shape == target.shape, "predict and target must have the same shape"

    num_classes = target.shape[1]
    count = 0
    dice_list = []
    for i in range(1, num_classes):
        _dice = 0.
        if torch.any(target[:, i]):
            count = count + 1
            _dice = dice_coefficient(predict[:, i], target[:, i]).item()
        dice_list.append(_dice)
    if reduction == "mean":
        res = sum(dice_list) / count if count > 0 else 0.
        return torch.tensor(res, device=predict.device)
    else:
        return torch.tensor(dice_list, device=predict.device)
