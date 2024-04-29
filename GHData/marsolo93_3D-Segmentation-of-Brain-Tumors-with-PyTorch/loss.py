import torch
from torch import nn
import config


class LossAndMetric(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.c_entropy = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        closs = self.c_entropy(pred, target)
        meaniou = self.weightedmeaniou(pred, target)
        lossiou = 1 - torch.mean(meaniou)
        loss = 50 * closs + lossiou
        return loss, meaniou, lossiou

    def weightedmeaniou(self, pred, target):
        pred = torch.sigmoid(pred)
        targetflat = self.flatten(target)
        w = targetflat.sum(-1)
        w = 1 / (w).clamp(min=1e-6)
        w.requires_grad = False
        intersection = torch.sum(torch.abs(pred * target), dim=[-3, -2, -1]) * w
        union = (torch.sum(pred**2 + target**2, dim=[-3, -2, -1]) * w)
        res = torch.mean((2 * intersection + 1e-2 ) / (union + 1e-2), dim=0)
        return res

    def meaniou(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(target.long(), num_classes=self.num_classes).permute([0, 4, 1, 2, 3])
        intersection = torch.sum(torch.abs(pred * target), dim=[-3, -2, -1])
        union = (torch.sum(pred + target, dim=[-3, -2, -1])) - intersection
        return torch.mean((intersection + 1) / (union + 1), dim=0)

    def gdice(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(target.long(), num_classes=self.num_classes).permute([0, 4, 1, 2, 3])
        pred = self.flatten(pred)
        target = self.flatten(target).float()
        w = target.sum(-1)
        w = 1 / (w).clamp(min=1e-6)
        w.requires_grad = False

        intersect = (pred * target).sum(-1)
        intersect = intersect * w

        denominator = (pred + target).sum(-1)
        denominator = (denominator * w).clamp(min=1e-6)
        return 1 - ( (2 * intersect.sum() + 1) / (denominator.sum() + 1) )

    def flatten(self, x):
        return torch.flatten(x, start_dim=-3, end_dim=-1)

if __name__ == '__main__':
    import numpy as np
    from utils import *

    csv_path = '/media/tensorist/Extreme SSD/brats2020/trainset.csv'

    loader = BrainLoader(csv_path)
    loss_metric = LossAndMetric(num_classes=3)
    for i in range(len(loader)):
        img, gt = loader[i]
        img = img.squeeze()
        gt = gt[np.newaxis, ...]

        loss, meaniou, iouloss = loss_metric(torch.tensor(gt), torch.tensor(gt))
        print('Loss:', loss)
        print('MeanIOU: ', meaniou)
        print('iouloss: ', iouloss)


