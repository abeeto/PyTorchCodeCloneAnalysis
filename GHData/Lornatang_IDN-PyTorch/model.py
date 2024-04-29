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
from torch.nn import functional as F

__all__ = [
    "IDN",
]


class _InformationDistillationLayer(nn.Module):
    def __init__(self, channels: int, channels_diff: int, slices: int) -> None:
        super(_InformationDistillationLayer, self).__init__()
        self.channels = channels
        self.channels_diff = channels_diff
        self.slices = slices

        self.slice_up_layers = nn.Sequential(
            nn.Conv2d(channels, channels - channels_diff, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(channels - channels_diff, channels - int(2 * channels_diff), (3, 3), (1, 1), (1, 1),
                      groups=slices),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(int(2 * channels_diff), channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
        )

        self.slice_down_layers = nn.Sequential(
            nn.Conv2d(channels - channels // slices, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(channels, channels - channels_diff, (3, 3), (1, 1), (1, 1), groups=slices),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(channels - channels_diff, channels + channels_diff, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
        )

        self.compress = nn.Conv2d(channels + channels_diff, channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slice_up_out = self.slice_up_layers(x)

        slice_up_out_feature1 = slice_up_out[:, 0:(self.channels - self.channels // self.slices), :, :]
        slice_up_out_feature2 = slice_up_out[:, (self.channels - self.channels // self.slices):self.channels, :, :]

        out = self.slice_down_layers(slice_up_out_feature1)
        concat = torch.cat((slice_up_out_feature2, x), 1)
        out = torch.add(out, concat)
        out = self.compress(out)

        return out


class IDN(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(IDN, self).__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.05, True),
        )

        self.idl1 = _InformationDistillationLayer(64, 16, 4)
        self.idl2 = _InformationDistillationLayer(64, 16, 4)
        self.idl3 = _InformationDistillationLayer(64, 16, 4)
        self.idl4 = _InformationDistillationLayer(64, 16, 4)

        self.upsample = nn.ConvTranspose2d(64, 3, (17, 17), (upscale_factor, upscale_factor), (8, 8))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode="bilinear")

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.idl1(out)
        out = self.idl2(out)
        out = self.idl3(out)
        out = self.idl4(out)

        out = self.upsample(out, output_size=bicubic.size())
        out = torch.add(out, bicubic)

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
