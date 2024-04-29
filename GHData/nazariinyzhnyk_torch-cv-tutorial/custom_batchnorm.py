import numpy as np
import torch
import torch.nn as nn

batch_size = 8
input_channels = 3
height = width = 128


def custom_batch_norm1d(input_tensor, eps, gamma=1, beta=0):
    sd = torch.std(input_tensor, axis=0, unbiased=False)
    mn = torch.mean(input_tensor, axis=0)
    normed_tensor = (input_tensor - mn) / (sd.pow(2) + eps).sqrt() * gamma + beta

    return normed_tensor


eps = np.power(10., -2)
input_tensor = torch.Tensor([[0.0, 0, 1, 0, 2], [0, 1, 1, 0, 10]])
batch_norm = nn.BatchNorm1d(input_tensor.shape[1], affine=False)
batch_norm.eps = eps
print(input_tensor.shape)
print(input_tensor[0])
print(batch_norm(input_tensor))
print(custom_batch_norm1d(input_tensor, eps))


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        self.test = False
        self.running_var = 1
        self.running_mean = 0

    def __call__(self, input_tensor):
        if self.test:
            normed_tensor = (input_tensor - self.running_mean) / (self.running_var + self.eps).sqrt() * self.weight + self.bias
            return normed_tensor
        else:
            sd = torch.std(input_tensor, dim=0, unbiased=False)
            var = sd.pow(2)
            mn = torch.mean(input_tensor, dim=0)
            normed_tensor = (input_tensor - mn) / (sd.pow(2) + self.eps).sqrt() * self.weight + self.bias
            self.running_var = (1-self.momentum) * var * batch_size / (batch_size-1) + self.momentum * self.running_var
            self.running_mean = (1-self.momentum) * mn + self.momentum * self.running_mean
            return normed_tensor

    def eval(self):
        self.test = True


def custom_batch_norm2d(input_tensor, eps):
    normed_tensor = input_tensor.reshape(batch_size, input_channels, height * width)
    for i in range(input_channels):
        sd = torch.std(normed_tensor[:, i, :], unbiased=False)
        mn = torch.mean(normed_tensor[:, i, :])
        normed_tensor[:, i, :] = (normed_tensor[:, i, :] - mn) / (sd.pow(2) + eps).sqrt()

    return normed_tensor.reshape(batch_size, input_channels, height, width)


def custom_instance_norm1d(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    for b in range(batch_size):
        for i in range(input_channels):
                sd = torch.std(input_tensor[b, i, :], dim=0, unbiased=False)
                mn = torch.mean(input_tensor[b, i, :], dim=0)
                normed_tensor[b, i, :] = (input_tensor[b, i, :] - mn) / (sd.pow(2) + eps).sqrt()
    return normed_tensor


def custom_layer_norm(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    for b in range(input_tensor.shape[0]):
        sd = torch.std(input_tensor[b], unbiased=False)
        mn = torch.mean(input_tensor[b])
        normed_tensor[b] = (input_tensor[b] - mn) / (sd.pow(2) + eps).sqrt()
    return normed_tensor


def custom_group_norm(input_tensor, groups, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    group_size = input_tensor.shape[1] / groups
    for b in range(input_tensor.shape[0]):
        for gs in range(groups):
            slc = int(gs * group_size)
            nslc = int((gs + 1) * group_size)
            sd = torch.std(input_tensor[b,slc:nslc,:], unbiased=False)
            mn = torch.mean(input_tensor[b,slc:nslc,:])
            normed_tensor[b,slc:nslc,:] = (input_tensor[b,slc:nslc,:] - mn) / (sd.pow(2) + eps).sqrt()
    return normed_tensor
