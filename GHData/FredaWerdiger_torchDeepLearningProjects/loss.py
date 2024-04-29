import torch
import torch.nn.functional as F

class diceloss(torch.nn.Module):
    '''
    https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
    https://github.com/pytorch/pytorch/issues/1249
    '''

    def __init__(self):
        super(diceloss, self).__init__()

    def forward(self, input, target):
        probs = F.softmax(input)
        num = probs * target  # b,c,h,w--p*g
        num = torch.sum(num, dim=3)  # b,c,h
        num = torch.sum(num, dim=2)

        den1 = probs * probs  # --p^2
        den1 = torch.sum(den1, dim=3)  # b,c,h
        den1 = torch.sum(den1, dim=2)

        den2 = target * target  # --g^2
        den2 = torch.sum(den2, dim=3)  # b,c,h
        den2 = torch.sum(den2, dim=2)  # b,c

        dice = 2 * (num / (den1 + den2))
        dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class diceloss1(torch.nn.Module):
    '''
    Adapted from https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py
    '''

    def __init__(self, smooth=1.):
        super(diceloss1, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        target = F.one_hot(target, num_classes=2).permute((0, 4, 1, 2, 3))
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))

        return loss