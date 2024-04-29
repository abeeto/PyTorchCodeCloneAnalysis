import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor != self.olabel] = 0
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        gray = torchvision.transforms.functional.to_grayscale(image, num_output_channels=1)
        return torch.from_numpy(np.array(gray)).long()


class Interpolate:
    def __init__(self, w, h):
        self.width =w
        self.height = h

    def __call__(self, image):
        return F.interpolate(image, size=(self.width, self.height))
