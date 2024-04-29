from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist

import errno
import os
###confusionmatrix for evaluation
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

##concate two tensors
def combine2tensor(tensor1, tensor2):
    n1, c1, h1, w1 = tensor1.shape
    n2, c2, h2, w2 = tensor2.shape
    assert h1==h2 and w1==w2, 'the shape of two tensor is different'
    tensor = torch.tensor((), dtype=torch.int32)
    tensor_combine = tensor.new_ones((n1, c1, 2*h1, w2 ))
    tensor_combine[:, :, 0:h2, :] = tensor1
    tensor_combine[:, :, h2:, : ] = tensor2 
    return tensor_combine