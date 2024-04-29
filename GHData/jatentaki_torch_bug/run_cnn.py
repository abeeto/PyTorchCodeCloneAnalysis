import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.misc import imread

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.cnv1 = nn.Conv3d(1, 2, kernel_size=(3, 7, 7),
                                     dilation=(1, 3, 3),
                                     padding=(1, 0, 0))
        self.max1 = nn.MaxPool3d(kernel_size=(1, 3, 3)  , padding=(1, 0, 0))

    def encode(self, x):
        y = self.cnv1(x)
        y = self.max1(y)

        return y

def to_cnn_layout(img):
	img = np.rollaxis(img, 1)
	img = np.rollaxis(img, 2)

	return img

def processing_stack(imgs):
	stack = np.stack([to_cnn_layout(i) for i in imgs])
	return torch.Tensor(stack[:, np.newaxis, :, :, :])

e1 = imread('bug.jpg')

stack = processing_stack([e1])

v = Variable(stack)
enc = Encoder()
y = enc.encode(v)