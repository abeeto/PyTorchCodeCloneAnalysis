import torch
from torch.nn import functional as F


def kl_loss(input_probs, target_probs):
    """
    Compute unreduced KL Divergence loss as : L = y_n.(log(y_n) - x_n)
    Sum over class dimension and then average over batch and spatial dimension
    """
    assert input_probs.size() == target_probs.size()
    return F.kl_div(input_probs, target_probs, reduction='none').sum(dim = 1).mean(dim = (0,1,2))

def robust_binary_crossentropy(pred, tgt, eps=1e-6):
    inv_tgt = 1.0 - tgt
    inv_pred = 1.0 - pred + eps
    return (-(tgt * torch.log(pred + eps) + inv_tgt * torch.log(inv_pred))).sum(dim = 1).mean(dim = (0,1,2))
