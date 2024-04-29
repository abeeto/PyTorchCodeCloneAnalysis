# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List, Type, Union

import torch
from torch import nn, Tensor

__all__ = [
    "PyramidNet",
    "pyramidnet18", "pyramidnet34", "pyramidnet50", "pyramidnet101", "pyramidnet152", "pyramidnet200",
]


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Sequential = None
    ) -> None:
        super(_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            feature_map_size = shortcut.size()[2:4]
        else:
            shortcut = x
            feature_map_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.FloatTensor(batch_size,
                                        residual_channel - shortcut_channel,
                                        feature_map_size[0],
                                        feature_map_size[1]).fill_(0)
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Sequential = None
    ) -> None:
        super(_Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, (out_channels * 1), (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d((out_channels * 1))
        self.conv3 = nn.Conv2d((out_channels * 1), out_channels * _Bottleneck.expansion, (1, 1), (1, 1),
                               (0, 0),
                               bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels * _Bottleneck.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            feature_map_size = shortcut.size()[2:4]
        else:
            shortcut = x
            feature_map_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.FloatTensor(batch_size,
                                        residual_channel - shortcut_channel,
                                        feature_map_size[0],
                                        feature_map_size[1]).fill_(0)
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class PyramidNet(nn.Module):

    def __init__(
            self,
            arch_cfg: List[int],
            block: Type[Union[_BasicBlock, _Bottleneck]],
            alpha: int,
            num_classes: int = 1000,
    ) -> None:
        super(PyramidNet, self).__init__()
        self.in_channels = 64
        self.increase_ratio = alpha / (sum(arch_cfg) * 1.0)

        self.input_feature_map_dim = self.in_channels
        self.conv1 = nn.Conv2d(3, self.input_feature_map_dim, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_feature_map_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.feature_map_dim = self.input_feature_map_dim
        self.layer1 = self._make_layer(arch_cfg[0], block, 1)
        self.layer2 = self._make_layer(arch_cfg[1], block, 2)
        self.layer3 = self._make_layer(arch_cfg[2], block, 2)
        self.layer4 = self._make_layer(arch_cfg[3], block, 2)

        self.final_feature_map_dim = self.input_feature_map_dim
        self.bn_final = nn.BatchNorm2d(self.final_feature_map_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.final_feature_map_dim, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(nn.AvgPool2d((2, 2), (2, 2), ceil_mode=True))

        self.feature_map_dim = self.feature_map_dim + self.increase_ratio
        layers = [
            block(
                self.input_feature_map_dim,
                int(round(self.feature_map_dim)),
                stride,
                downsample
            )
        ]
        for i in range(1, repeat_times):
            temp_feature_map_dim = self.feature_map_dim + self.increase_ratio
            layers.append(
                block(
                    int(round(self.feature_map_dim)) * block.expansion,
                    int(round(temp_feature_map_dim)),
                    1,
                    None
                )
            )
            self.feature_map_dim = temp_feature_map_dim

        self.input_feature_map_dim = int(round(self.feature_map_dim)) * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.bn_final(out)
        out = self.relu_final(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def pyramidnet18(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([2, 2, 2, 2], _BasicBlock, **kwargs)

    return model


def pyramidnet34(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([3, 4, 6, 3], _BasicBlock, **kwargs)

    return model


def pyramidnet50(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([3, 4, 6, 3], _Bottleneck, **kwargs)

    return model


def pyramidnet101(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([3, 4, 23, 3], _Bottleneck, **kwargs)

    return model


def pyramidnet152(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([3, 8, 36, 3], _Bottleneck, **kwargs)

    return model


def pyramidnet200(**kwargs: Any) -> PyramidNet:
    model = PyramidNet([3, 24, 36, 3], _Bottleneck, **kwargs)

    return model
