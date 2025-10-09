import monai
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


def dice_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """

    assert predict.shape == target.shape, "predict and target must have the same shape"

    num_classes = target.shape[1]
    total_dice = 0.
    count = 0

    for i in range(1, num_classes):
        if torch.any(target[:, i]):
            count = count + 1
            total_dice = total_dice + dice_coefficient(predict[:, i], target[:, i])

    return total_dice / count if count > 0 else torch.tensor(0., device=predict.device)


def dice_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_dice = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(target_i):
            count = count + 1
            total_dice = total_dice + dice_coefficient(predict_i, target_i)
    return total_dice / count if count > 0 else torch.tensor(0., device=predict.device)


def dice_metric3(predict, target, num_classes, reduction="mean"):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    # test
    predict = torch.tensor(predict)
    target = torch.tensor(target)
    
    count = 0
    dice_list = []
    for i in range(1, num_classes + 1):
        _dice = torch.tensor(0., device=predict.device)
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(target_i):
            count = count + 1
            _dice = dice_coefficient(predict_i, target_i)
        dice_list.append(_dice)

    if reduction == "mean":
        res = sum(dice_list) / count if count > 0 else 0.
        return torch.tensor(res, device=predict.device)
    else:
        return torch.tensor(dice_list, device=predict.device)


def ASD_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """
    num_classes = target.shape[1]
    total_ASD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = predict[:, i].unsqueeze(dim=1)
        target_i = target[:, i].unsqueeze(dim=1)
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            ASD_i = monai.metrics.compute_average_surface_distance(predict_i, target_i, include_background=True)
            total_ASD = total_ASD + torch.mean(ASD_i)

    return total_ASD / count if count > 0 else torch.inf


def ASD_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_ASD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            ASD_i = monai.metrics.compute_average_surface_distance(predict_i, target_i, include_background=True)
            total_ASD = total_ASD + torch.mean(ASD_i)
    return total_ASD / count if count > 0 else torch.inf


def ASD_metric3(predict, target, num_classes, reduction="mean"):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    new_axis = (0, -1) + tuple(range(1, target.ndim - 1))
    predict_one_hot = F.one_hot(predict.squeeze(dim=1), num_classes).permute(new_axis).contiguous()
    target_one_hot = F.one_hot(target.squeeze(dim=1), num_classes).permute(new_axis).contiguous()
    asd = monai.metrics.compute_average_surface_distance(predict_one_hot, target_one_hot, include_background=False)
    asd = torch.mean(asd, dim=0)
    if reduction == "mean":
        asd = torch.mean(asd)
    return asd


def HD_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """
    num_classes = target.shape[1]
    total_HD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = predict[:, i].unsqueeze(dim=1)
        target_i = target[:, i].unsqueeze(dim=1)
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            HD_i = monai.metrics.compute_hausdorff_distance(predict_i, target_i, include_background=True)
            total_HD = total_HD + torch.mean(HD_i)
    return total_HD / count if count > 0 else torch.inf


def HD_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_HD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            HD_i = monai.metrics.compute_hausdorff_distance(predict_i, target_i, include_background=True)
            total_HD = total_HD + torch.mean(HD_i)
    return total_HD / count if count > 0 else torch.inf


def HD_metric3(predict, target, num_classes, reduction="mean"):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    new_axis = (0, -1) + tuple(range(1, target.ndim - 1))
    predict_one_hot = F.one_hot(predict.squeeze(dim=1), num_classes).permute(new_axis).contiguous()
    target_one_hot = F.one_hot(target.squeeze(dim=1), num_classes).permute(new_axis).contiguous()
    hd = monai.metrics.compute_hausdorff_distance(predict_one_hot, target_one_hot, include_background=False)
    hd = torch.mean(hd, dim=0)
    if reduction == "mean":
        hd = torch.mean(hd)
    return hd


if __name__ == '__main__':
    import torch.nn.functional as F

    batch_size = 1
    num_classes = 4
    a = torch.randint(0, num_classes, size=(batch_size, 1, 128, 128, 128))
    b = torch.randint(0, num_classes, size=(batch_size, 1, 128, 128, 128))

    new_axis = (0, -1) + tuple(range(1, a.ndim - 1))
    a_one_hot = F.one_hot(a.squeeze(dim=1), num_classes).permute((0, -1, 1, 2, 3)).contiguous()
    b_one_hot = F.one_hot(b.squeeze(dim=1), num_classes).permute((0, -1, 1, 2, 3)).contiguous()
    # print(a_one_hot.shape)

    asd = monai.metrics.compute_average_surface_distance(a_one_hot, b_one_hot, include_background=False)
    # asd_1 = ASD_metric2(a, b, 3)
    asd_1 = ASD_metric3(a, b, num_classes,reduction="none")
    hd_1 = HD_metric3(a, b, num_classes,reduction="none")
    print(asd_1)
    # print(torch.mean(asd))
    print(hd_1)

    # output1 = monai.metrics.compute_generalized_dice(a_one_hot, b_one_hot, include_background=False)
    #
    # output2 = dice_metric3(a, b, num_classes=num_classes - 1, reduction="")
    #
    # print(output1)
    # print(output2)
    # print(torch.mean(output1))
    # print(torch.mean(output2))
