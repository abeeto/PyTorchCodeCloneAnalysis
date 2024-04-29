import numpy as np
import torch
import torch.nn.functional as F


def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True:
        labels = y_true_in
        y_pred = y_pred_in
        true_objects = 2
        pred_objects = 2
        # 重合区域
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # 计算区域
        area_true = np.histogram(labels, bins=true_objects)[0]
        area_pred = np.histogram(y_pred, bins=pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # union
        union = area_true + area_pred - intersection

        # intersection over union
        intersection = intersection[1:, 1:]
        union = union[1:, 1:]
        union[union == 0] = 1e-9

        iou = intersection / union

        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1
            false_positives = np.sum(matches, axis=0) == 0
            false_negatives = np.sum(matches, axis=1) == 0
            true_positive, false_positive, false_negative = np.sum(true_positives), np.sum(false_positives), np.sum(
                false_negatives)
            return true_positive, false_positive, false_negative

        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0


def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:, 0, :, :]
    target = target[:, 0, :, :]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)


def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)


def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    intersection = (output * target).sum()

    return (intersection + smooth) / (target.sum() + smooth)
