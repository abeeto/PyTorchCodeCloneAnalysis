from torch import nn
import torch


def CrossEntropyLoss(ignore_index=None):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def WeightCrossEntropyLoss(weight=None, ignore_index=-100):
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

class DiceLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target, smooth=1e-7):
        """
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to
                       the raw output or logits of the model.
        :param target: a tensor of shape [B, H, W].
        :param smooth: added to the denominator for numerical stability.
        :return:
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            target_oh = torch.eye(num_classes + 1)[target]
            target_oh = target_oh.permute(0, 3, 1, 2).float()
            target_oh_f = target_oh[:, 0:1, :, :]
            target_oh_s = target_oh[:, 1:2, :, :]
            target_oh = torch.cat([target_oh_s, target_oh_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            target_oh = torch.eye(num_classes)[target]  # 可以直接根据target的数值转换成为one_hot编码
            target_oh = target_oh.permute(0, 3, 1, 2).float()
            probas = nn.Softmax(logits,dim=1)
        target_oh = target_oh.type(logits.type())
        dims = (0,) + tuple(range(2, target.ndimension()+1))  # ensuring the dims for sum
        intersection = torch.sum(probas * target_oh, dims)
        cardinality = torch.sum(probas + target_oh, dims)
        dice_loss = (2. * intersection / (cardinality + smooth)).mean()
        return (1 - dice_loss)
