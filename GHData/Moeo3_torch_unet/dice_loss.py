import torch
from torch.nn import Module

class DiceLoss(Module):
    def __init__(self, eps = 1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        N = y_pred.shape[0]
        y_true_f = y_true.view(N, -1)
        y_pred_f = y_pred.view(N, -1)
        tp = torch.sum(y_true_f * y_pred_f, dim=1)
        fp = torch.sum(y_pred_f, dim=1) - tp
        fn = torch.sum(y_true_f, dim=1) - tp
        dice_coef = (2. * tp + self.eps) / (2. * tp + fp + fn + self.eps)
        return torch.mean(1. - dice_coef)