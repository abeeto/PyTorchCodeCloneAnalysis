import torch
from torch import nn
import numpy as np


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(out_features, in_features))
        std = np.sqrt(variance_scale / in_features)

        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            scale = std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)
            scale = 1.0

        if bias:
            bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.scale = scale
        self.bias = bias

    def forward(self, inputs):

        outputs = nn.functional.linear(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias
        )

        return outputs


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        std = np.sqrt(variance_scale / num_embeddings)

        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            scale = std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)
            scale = 1.0

        self.weight = weight
        self.scale = scale

    def forward(self, inputs):

        outputs = nn.functional.embedding(
            input=inputs,
            weight=self.weight * self.scale
        )

        return outputs


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)

        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            scale = std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)
            scale = 1.0

        if bias:
            bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, inputs):

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )

        return outputs


class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, variance_scale, weight_scale):

        super().__init__()

        weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        std = np.sqrt(variance_scale / in_channels / kernel_size / kernel_size)

        if weight_scale:
            nn.init.normal_(weight, mean=0.0, std=1.0)
            scale = std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)
            scale = 1.0

        if bias:
            bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(bias)
        else:
            bias = None

        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, inputs):

        outputs = nn.functional.conv_transpose2d(
            input=inputs,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )

        return outputs


class PixelNorm(nn.Module):

    def __init__(self, epsilon=1e-12):

        super().__init__()

        self.epsilon = epsilon

    def forward(self, inputs):

        norm = torch.mean(inputs ** 2, dim=1, keepdim=True)
        norm = torch.sqrt(norm + self.epsilon)
        outputs = inputs / norm

        return outputs


class BatchStd(nn.Module):

    def __init__(self, groups, epsilon=1e-12):

        super().__init__()

        self.groups = groups
        self.epsilon = epsilon

    def forward(self, inputs):

        outputs = inputs.reshape(self.groups, -1, *inputs.shape[1:])
        outputs -= torch.mean(outputs, dim=0, keepdim=True)
        outputs = torch.mean(outputs ** 2, dim=0)
        outputs = torch.sqrt(outputs + self.epsilon)
        outputs = torch.mean(outputs, dim=(1, 2, 3), keepdim=True)
        outputs = outputs.repeat(self.groups, 1, *inputs.shape[2:])

        return outputs


class LearnedConstant(nn.Module):

    def __init__(self, num_channels, resolution):

        super().__init__()

        self.constant = nn.Parameter(torch.ones(1, num_channels, resolution, resolution))

    def forward(self, inputs):

        outputs = self.constant.repeat(inputs.shape[0], *(1 for _ in self.constant.shape[1:]))

        return outputs


class LearnedNoise(nn.Module):

    def __init__(self, num_channels):

        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, inputs):

        noises = torch.randn(inputs.shape[0], 1, *inputs.shape[2:]).to(inputs.device)
        outputs = inputs + noises * self.weight

        return outputs


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, num_features, num_channels, bias, variance_scale, weight_scale):

        super().__init__()

        self.instance_norm2d = nn.InstanceNorm2d(
            num_features=num_channels,
            affine=False
        )

        self.linear1 = Linear(
            in_features=num_features,
            out_features=num_channels,
            bias=bias,
            variance_scale=variance_scale,
            weight_scale=weight_scale
        )

        self.linear2 = Linear(
            in_features=num_features,
            out_features=num_channels,
            bias=bias,
            variance_scale=variance_scale,
            weight_scale=weight_scale
        )

    def forward(self, inputs, styles):

        outputs = self.instance_norm2d(inputs)

        gamma = self.linear1(styles)
        beta = self.linear2(styles)

        outputs *= gamma.unsqueeze(-1).unsqueeze(-1)
        outputs += beta.unsqueeze(-1).unsqueeze(-1)

        return outputs
