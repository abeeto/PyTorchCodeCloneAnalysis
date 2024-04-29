import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim


class ModelBase(nn.Module):
    def __init__(self, image_size: int, lr: float = 0.1):
        super(ModelBase, self).__init__()
        self.image_size = image_size
        self.lr = lr

    @property
    def optimizer(self) -> optim.Optimizer:
        raise NotImplementedError

    @property
    def loss_function(self):
        return F.nll_loss

    def forward(self, data: Tensor):
        raise NotImplementedError
