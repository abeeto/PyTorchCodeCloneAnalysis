import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.0, gamma=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        ce = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = 1.0
        modulation_factor = 1.0

        if self.alpha:
            alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        if self.gamma:
            modulation_factor = (1.0 - p_t) ** self.gamma
        focal_loss = torch.sum(alpha_factor * modulation_factor * ce, 1)

        return torch.mean(focal_loss)