import torch.nn.functional as F
import torch.nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropyLoss():
    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func