import torch
from torch import nn
import numpy as np
import functools
from collections import OrderedDict
from modules import *


class MappingNetwork(nn.Module):

    def __init__(self, embedding_param, linear_params):

        super().__init__()

        self.module_dict = nn.ModuleDict(OrderedDict(
            embedding_block=nn.ModuleDict(OrderedDict(
                embedding=Embedding(
                    num_embeddings=embedding_param.num_embeddings,
                    embedding_dim=embedding_param.embedding_dim,
                    variance_scale=1,
                    weight_scale=True
                ),
                pixel_norm=PixelNorm()
            )),
            linear_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    linear=Linear(
                        in_features=linear_param.in_features,
                        out_features=linear_param.out_features,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU(0.2)
                ))
                for linear_param in linear_params
            ])
        ))

    def forward(self, latents, labels=None):

        if labels is not None:
            labels = self.module_dict.embedding_block.embedding(labels)
            latents = torch.cat((latents, labels), dim=1)

        latents = self.module_dict.embedding_block.pixel_norm(latents)

        for linear_block in self.module_dict.linear_blocks:
            latents = linear_block.linear(latents)
            latents = linear_block.leaky_relu(latents)

        return latents


class Generator(nn.Module):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, num_features, out_channels):

        super().__init__()

        min_depth = int(np.log2(min_resolution // min_resolution))
        max_depth = int(np.log2(max_resolution // min_resolution))

        def resolution(depth): return min_resolution << depth
        def num_channels(depth): return min(max_channels, min_channels << (max_depth - depth))

        self.module_dict = nn.ModuleDict(OrderedDict(
            conv_block=nn.ModuleDict(OrderedDict(
                first=nn.ModuleDict(OrderedDict(
                    leaned_constant=LearnedConstant(
                        num_channels=num_channels(min_depth),
                        resolution=resolution(min_depth)
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(min_depth)
                    ),
                    leaky_relu=nn.LeakyReLU(0.2),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        num_features=num_features,
                        num_channels=num_channels(min_depth),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                )),
                second=nn.ModuleDict(OrderedDict(
                    conv2d=Conv2d(
                        in_channels=num_channels(min_depth),
                        out_channels=num_channels(min_depth),
                        kernel_size=3,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    learned_noise=LearnedNoise(
                        num_channels=num_channels(min_depth)
                    ),
                    leaky_relu=nn.LeakyReLU(0.2),
                    adaptive_instance_norm=AdaptiveInstanceNorm(
                        num_features=num_features,
                        num_channels=num_channels(min_depth),
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                ))
            )),
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    first=nn.ModuleDict(OrderedDict(
                        conv_transpose2d=ConvTranspose2d(
                            in_channels=num_channels(depth - 1),
                            out_channels=num_channels(depth),
                            kernel_size=3,
                            stride=2,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(depth)
                        ),
                        leaky_relu=nn.LeakyReLU(0.2),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            num_features=num_features,
                            num_channels=num_channels(depth),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    )),
                    second=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(depth),
                            out_channels=num_channels(depth),
                            kernel_size=3,
                            stride=1,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        learned_noise=LearnedNoise(
                            num_channels=num_channels(depth)
                        ),
                        leaky_relu=nn.LeakyReLU(0.2),
                        adaptive_instance_norm=AdaptiveInstanceNorm(
                            num_features=num_features,
                            num_channels=num_channels(depth),
                            bias=True,
                            variance_scale=1,
                            weight_scale=True
                        )
                    ))
                )) for depth in range(min_depth + 1, max_depth + 1)
            ]),
            color_block=nn.ModuleDict(OrderedDict(
                conv2d=Conv2d(
                    in_channels=num_channels(max_depth),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    variance_scale=1,
                    weight_scale=True
                ),
                tanh=nn.Tanh()
            ))
        ))

    def forward(self, latents):

        outputs = self.module_dict.conv_block.first.leaned_constant(latents)
        outputs = self.module_dict.conv_block.first.learned_noise(outputs)
        outputs = self.module_dict.conv_block.first.leaky_relu(outputs)
        outputs = self.module_dict.conv_block.first.adaptive_instance_norm(outputs, latents)

        outputs = self.module_dict.conv_block.second.conv2d(outputs)
        outputs = self.module_dict.conv_block.second.learned_noise(outputs)
        outputs = self.module_dict.conv_block.second.leaky_relu(outputs)
        outputs = self.module_dict.conv_block.second.adaptive_instance_norm(outputs, latents)

        for conv_block in self.module_dict.conv_blocks:

            outputs = conv_block.first.conv_transpose2d(outputs)
            outputs = conv_block.first.learned_noise(outputs)
            outputs = conv_block.first.leaky_relu(outputs)
            outputs = conv_block.first.adaptive_instance_norm(outputs, latents)

            outputs = conv_block.second.conv2d(outputs)
            outputs = conv_block.second.learned_noise(outputs)
            outputs = conv_block.second.leaky_relu(outputs)
            outputs = conv_block.second.adaptive_instance_norm(outputs, latents)

        outputs = self.module_dict.color_block.conv2d(outputs)
        # outputs = self.module_dict.color_block.tanh(outputs)

        return outputs


class Discriminator(nn.Module):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, num_classes, in_channels):

        super().__init__()

        min_depth = int(np.log2(min_resolution // min_resolution))
        max_depth = int(np.log2(max_resolution // min_resolution))

        def resolution(depth): return min_resolution << depth
        def num_channels(depth): return min(max_channels, min_channels << (max_depth - depth))

        self.module_dict = nn.ModuleDict(OrderedDict(
            color_block=nn.ModuleDict(OrderedDict(
                conv2d=Conv2d(
                    in_channels=in_channels,
                    out_channels=num_channels(max_depth),
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    variance_scale=2,
                    weight_scale=True
                ),
                leaky_relu=nn.LeakyReLU(0.2)
            )),
            conv_blocks=nn.ModuleList([
                nn.ModuleDict(OrderedDict(
                    first=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(depth),
                            out_channels=num_channels(depth),
                            kernel_size=3,
                            stride=1,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        leaky_relu=nn.LeakyReLU(0.2)
                    )),
                    second=nn.ModuleDict(OrderedDict(
                        conv2d=Conv2d(
                            in_channels=num_channels(depth),
                            out_channels=num_channels(depth - 1),
                            kernel_size=3,
                            stride=2,
                            bias=True,
                            variance_scale=2,
                            weight_scale=True
                        ),
                        leaky_relu=nn.LeakyReLU(0.2)
                    ))
                )) for depth in range(max_depth, min_depth, -1)
            ]),
            conv_block=nn.ModuleDict(OrderedDict(
                first=nn.ModuleDict(OrderedDict(
                    batch_std=BatchStd(groups=4),
                    conv2d=Conv2d(
                        in_channels=num_channels(min_depth) + 1,
                        out_channels=num_channels(min_depth),
                        kernel_size=3,
                        stride=1,
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU(0.2)
                )),
                second=nn.ModuleDict(OrderedDict(
                    linear=Linear(
                        in_features=num_channels(min_depth) * min_resolution ** 2,
                        out_features=num_channels(min_depth - 1),
                        bias=True,
                        variance_scale=2,
                        weight_scale=True
                    ),
                    leaky_relu=nn.LeakyReLU(0.2)
                )),
                third=nn.ModuleDict(OrderedDict(
                    linear=Linear(
                        in_features=num_channels(min_depth - 1),
                        out_features=num_classes,
                        bias=True,
                        variance_scale=1,
                        weight_scale=True
                    )
                ))
            ))
        ))

    def forward(self, images, labels=None):

        outputs = self.module_dict.color_block.conv2d(images)
        outputs = self.module_dict.color_block.leaky_relu(outputs)

        for conv_block in self.module_dict.conv_blocks:

            outputs = conv_block.first.conv2d(outputs)
            outputs = conv_block.first.leaky_relu(outputs)

            outputs = conv_block.second.conv2d(outputs)
            outputs = conv_block.second.leaky_relu(outputs)

        outputs = torch.cat((outputs, self.module_dict.conv_block.first.batch_std(outputs)), dim=1)
        outputs = self.module_dict.conv_block.first.conv2d(outputs)
        outputs = self.module_dict.conv_block.first.leaky_relu(outputs)

        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.module_dict.conv_block.second.linear(outputs)
        outputs = self.module_dict.conv_block.second.leaky_relu(outputs)

        outputs = self.module_dict.conv_block.third.linear(outputs)

        if labels is not None:
            outputs = torch.gather(outputs, dim=1, index=labels.unsqueeze(-1))
            outputs = outputs.squeeze(-1)

        return outputs
