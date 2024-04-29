import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """
    IoULoss
    """

    def __init__(self, pos_weight, neg_weight, pos_or_neg=True, ignore_empty=True, ignore_full=True, reduction='none', verbose=False):
        super(IoULoss, self).__init__()
        if verbose:
            print('IoULoss pos_weight:', pos_weight)
            print('IoULoss neg_weight:', neg_weight)
            print('IoULoss reduction:', reduction)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.pos_or_neg = pos_or_neg
        self.ignore_empty = ignore_empty
        self.ignore_full = ignore_full
        self.reduction = reduction
        

    def __call__(self, input, target):
        th_input = input.sigmoid()
        th_target = target

        intersection_p = torch.sum(th_input * th_target, (2,3))
        union_p = torch.sum(th_input + th_target - th_input * th_target, (2,3))# + 1e-8
        
        intersection_n = torch.sum((1-th_input) * (1-th_target), (2,3))
        union_n = torch.sum((1-th_input) + (1-th_target) - (1-th_input) * (1-th_target), (2,3))# + 1e-8

        if self.pos_or_neg:
            pos = (torch.amax(th_target, (2,3)) > 0.5).float()
            iou_loss = self.pos_weight.unsqueeze(0) * pos * ((1 - intersection_p / union_p).clamp_(0,1)) + self.neg_weight.unsqueeze(0) * (1-pos) * ((1 - intersection_n / union_n).clamp_(0,1))
        else:
            pos = (torch.amax(th_target, (2,3)) > 0.5).float() if self.ignore_empty else torch.ones_like(intersection_p)
            neg = (torch.amin(th_target, (2,3)) < 0.5).float() if self.ignore_full else torch.ones_like(intersection_n)
            iou_loss = torch.zeros(pos.shape).cuda()
            iou_loss = self.pos_weight.unsqueeze(0) * pos * ((1 - intersection_p / union_p).clamp_(0,1)) + self.neg_weight.unsqueeze(0) * neg * ((1 - intersection_n / union_n).clamp_(0,1))

        if self.reduction == 'none':
            final_loss = iou_loss
        elif self.reduction == 'mean':
            final_loss = iou_loss.mean()
        elif self.reduction == 'sum':
            final_loss = iou_loss.sum()
        else:
            final_loss = iou_loss
        
        return final_loss
