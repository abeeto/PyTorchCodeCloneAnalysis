import torch as t
import torch.nn as nn
from models.VGG16 import VGG16 as vgg


class Fast_RCNN(nn.Module):
    def __init__(self):
        super(Fast_RCNN, self).__init__()
        self.vgg = vgg
        self.roi_pooling = nn.AdaptiveMaxPool2d(7)

    def forward(self, x):
        output = self.vgg(x)
        # TODO
