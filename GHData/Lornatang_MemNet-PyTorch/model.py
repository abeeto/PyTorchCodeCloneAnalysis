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
import torch
from torch import nn

__all__ = [
    "MemNet",
]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identify = x
        out = self.residual_block(x)
        out = torch.add(out, identify)

        return out


class _MemoryBlock(nn.Module):
    def __init__(self, channels: int, num_memory_blocks: int, num_residual_blocks: int) -> None:
        super(_MemoryBlock, self).__init__()
        gate_channels = int((num_residual_blocks + num_memory_blocks) * channels)
        self.num_residual_blocks = num_residual_blocks

        recursive_unit = []
        for _ in range(num_residual_blocks):
            recursive_unit.append(_ResidualBlock(channels))
        self.recursive_unit = nn.Sequential(*recursive_unit)

        self.gate_unit = nn.Sequential(
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(True),
            nn.Conv2d(gate_channels, channels, (1, 1), (1, 1), (0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor, long_outs: list) -> torch.Tensor:
        out = x

        short_outs = []
        for _ in range(self.num_residual_blocks):
            out = self.recursive_unit(out)
            short_outs.append(out)

        gate_out = self.gate_unit(torch.cat(short_outs + long_outs, 1))
        long_outs.append(gate_out)

        return gate_out


class MemNet(nn.Module):
    def __init__(self, num_memory_blocks: int, num_residual_blocks: int) -> None:
        super(MemNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
        )

        dense_memory_blocks = []
        for i in range(num_memory_blocks):
            dense_memory_blocks.append(_MemoryBlock(64, i + 1, num_residual_blocks))
        self.dense_memory_blocks = nn.Sequential(*dense_memory_blocks)

        self.reconstructor = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, (1, 1), (1, 1), (0, 0), bias=False),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identify = x

        out = self.feature_extractor(x)

        long_outs = [out]
        for memory_block in self.dense_memory_blocks:
            out = memory_block(out, long_outs)

        out = self.reconstructor(out)

        out = torch.add(out, identify)
        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
