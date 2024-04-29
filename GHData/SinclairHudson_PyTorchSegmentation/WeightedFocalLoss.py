from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import time
from torch.autograd import Variable


def one_hot(index, classes):
    final = np.zeros((index.size(0), classes, index.size(2), index.size(3)))
    tar = index.cpu().numpy()

class FocalLoss(nn.Module):
    """
    This is weighted focal loss, first described in https://arxiv.org/pdf/1708.02002.pdf
    Essentially, this loss is like crossentropy, except it ignores classes that the network confidently gets correct.
    It focuses more on the challenging examples, which usually makes the network learn faster, and also helps with 
    class imbalances.
    The gamma element controls how much the loss focuses on the harder examples.
    When gamma=0, this module behaves identically to nn.NLLLoss.
    """
    def __init__(self, gamma=0.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        """
        The forward computation of loss
        :param input: [N x C x -1], containing log-probabilities of each class
        :param target: [N x -1], containing the correct labels. These should be ints in range [0, C -1]
        :return: the loss, either as a mean or a total. it depends on the reduction
        """
        pt = torch.exp(input)  # so now we're right back to probabilities
        focal_log_probabilities = ((1 - pt) ** self.gamma) * torch.log(pt)
        # at this point, focal_log_probabilities is a tensor containing focal log probabilities as if every class was
        # the correct one
        # now we just have to sum, keeping only the predictions for the correct class.
        # think of nll_loss as a fancy summation with a negative in front - that's all it does.
        return F.nll_loss(focal_log_probabilities, target, weight=self.weight)
        
# Testing
if __name__ == "__main__":
    w = torch.Tensor([40, 30, 2, 24])
    baseline = nn.NLLLoss(weight=w)
    crit = FocalLoss(gamma=2, weight=w)

    label = torch.randint(4, (3, 64, 512))
    x = 3 * torch.rand((3, 4, 64, 512))

    input = torch.log(F.softmax(x, dim=1))  # it's important to log softmax before
    loss = crit(input, label)
    base = baseline(input, label)
    print(F.nll_loss(input, label, weight=w, reduce=None, reduction="mean"))
    print(loss)
    print(base)

