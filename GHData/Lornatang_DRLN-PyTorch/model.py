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
import math

import torch
from torch import nn

__all__ = [
    "LaplacianAttentionLayer", "ResidualLayer", "ResidualBlock", "UpsampleBlock",
    "DRLN",
]


class LaplacianAttentionLayer(nn.Module):
    """Implements `Laplacian attention`.
    Which has a two-fold purpose:
        1) To learn the features at multiple sub-band frequencies
        2) to adaptively rescale features and model feature dependencies.
    Laplacian attention further improves the feature capturing capability of our network.

    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super(LaplacianAttentionLayer, self).__init__()
        reduction_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, (3, 3), (1, 1), (3, 3), (3, 3)),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, (3, 3), (1, 1), (5, 5), (5, 5)),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, (3, 3), (1, 1), (7, 7), (7, 7)),
            nn.ReLU(True),
        )
        self.conv4_sigmoid = nn.Sequential(
            nn.Conv2d(int(reduction_channels * 3), channels, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool_out = self.avg_pool(x)

        conv1_out = self.conv1(avg_pool_out)
        conv2_out = self.conv2(avg_pool_out)
        conv3_out = self.conv3(avg_pool_out)
        cat_conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)

        # Calculate the global area information strength
        out = self.conv4_sigmoid(cat_conv_out)
        out = torch.mul(x, out)

        return out


class ResidualLayer(nn.Module):
    """Implements `Residual Laplacian Module`.
    Each cascading block has a medium skip-connection (MSC), cascading features concatenation
    and is made up of dense residual Laplacian modules (DRLN) each of which consists of a densely
    connected residual unit, compression unit and Laplacian pyramid attention unit

    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualLayer, self).__init__()
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_layer(x)
        out = torch.add(x, out)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    """Implements `Dense Residual Laplacian Module`.
    Each cascading block has a medium skip-connection (MSC), cascading features concatenation
    and is made up of dense residual Laplacian modules (DRLN) each of which consists of a densely
    connected residual unit, compression unit and Laplacian pyramid attention unit

    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.residual_layer1 = ResidualLayer(in_channels, out_channels)
        self.residual_layer2 = ResidualLayer(in_channels * 2, out_channels * 2)
        self.residual_layer3 = ResidualLayer(in_channels * 4, out_channels * 4)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels * 8, out_channels, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
        )

        self.laplacian_attention_layer = LaplacianAttentionLayer(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_layer1 = self.residual_layer1(x)
        out1 = torch.cat([x, residual_layer1], 1)

        residual_layer2 = self.residual_layer2(out1)
        cat_out2 = torch.cat([out1, residual_layer2], 1)

        residual_layer3 = self.residual_layer3(cat_out2)
        cat_out3 = torch.cat([cat_out2, residual_layer3], 1)

        out = self.conv_layer(cat_out3)
        out = self.laplacian_attention_layer(out)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class DRLN(nn.Module):
    """Implements `Densely Residual Laplacian Network`."""

    def __init__(self, upscale_factor: int) -> None:
        super(DRLN, self).__init__()
        # First layer
        self.head = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone layer
        self.rb1 = ResidualBlock(64, 64)
        self.rb2 = ResidualBlock(64, 64)
        self.rb3 = ResidualBlock(64, 64)
        self.rb4 = ResidualBlock(64, 64)
        self.rb5 = ResidualBlock(64, 64)
        self.rb6 = ResidualBlock(64, 64)
        self.rb7 = ResidualBlock(64, 64)
        self.rb8 = ResidualBlock(64, 64)
        self.rb9 = ResidualBlock(64, 64)
        self.rb10 = ResidualBlock(64, 64)
        self.rb11 = ResidualBlock(64, 64)
        self.rb12 = ResidualBlock(64, 64)

        self.rb13 = ResidualBlock(64, 64)
        self.rb14 = ResidualBlock(64, 64)
        self.rb15 = ResidualBlock(64, 64)
        self.rb16 = ResidualBlock(64, 64)
        self.rb17 = ResidualBlock(64, 64)
        self.rb18 = ResidualBlock(64, 64)
        self.rb19 = ResidualBlock(64, 64)
        self.rb20 = ResidualBlock(64, 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(320, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        self.conv17 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv19 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )
        self.conv20 = nn.Sequential(
            nn.Conv2d(320, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.tail = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        x = x.sub_(self.mean).mul_(255.)

        out = self.head(x)

        # The first short skip connections(SSC) module
        rb1 = self.rb1(out)
        cat1 = torch.cat([out, rb1], 1)
        conv1 = self.conv1(cat1)
        rb2 = self.rb2(conv1)
        cat2 = torch.cat([cat1, rb2], 1)
        conv2 = self.conv2(cat2)
        rb3 = self.rb3(conv2)
        cat3 = torch.cat([cat2, rb3], 1)
        conv3 = self.conv3(cat3)
        out1 = torch.add(conv3, out)

        # The second short skip connections(SSC) module
        rb4 = self.rb4(out1)
        cat4 = torch.cat([conv3, rb4], 1)
        conv4 = self.conv4(cat4)
        rb5 = self.rb5(conv4)
        cat5 = torch.cat([cat4, rb5], 1)
        conv5 = self.conv5(cat5)
        rb6 = self.rb6(conv5)
        cat6 = torch.cat([cat5, rb6], 1)
        conv6 = self.conv6(cat6)
        out2 = torch.add(conv6, out1)

        # The third short skip connections(SSC) module
        rb7 = self.rb7(out2)
        cat7 = torch.cat([conv6, rb7], 1)
        conv7 = self.conv7(cat7)
        rb8 = self.rb8(conv7)
        cat8 = torch.cat([cat7, rb8], 1)
        conv8 = self.conv8(cat8)
        rb9 = self.rb9(conv8)
        cat9 = torch.cat([cat8, rb9], 1)
        conv9 = self.conv9(cat9)
        out3 = torch.add(conv9, out2)

        # The fourth short skip connections(SSC) module
        rb10 = self.rb10(out3)
        cat10 = torch.cat([conv9, rb10], 1)
        conv10 = self.conv10(cat10)
        rb11 = self.rb11(conv10)
        c11 = torch.cat([cat10, rb11], 1)
        conv11 = self.conv11(c11)
        rb12 = self.rb12(conv11)
        c12 = torch.cat([c11, rb12], 1)
        conv12 = self.conv12(c12)
        out4 = torch.add(conv12, out3)

        # The fifth short skip connections(SSC) module
        rb13 = self.rb13(out4)
        cat13 = torch.cat([conv12, rb13], 1)
        conv13 = self.conv13(cat13)
        rb14 = self.rb14(conv13)
        cat14 = torch.cat([cat13, rb14], 1)
        conv14 = self.conv14(cat14)
        rb15 = self.rb15(conv14)
        cat15 = torch.cat([cat14, rb15], 1)
        conv15 = self.conv15(cat15)
        rb16 = self.rb16(conv15)
        cat16 = torch.cat([cat15, rb16], 1)
        conv16 = self.conv16(cat16)
        out5 = torch.add(conv16, out4)

        # The sixth short skip connections(SSC) module
        rb17 = self.rb17(out5)
        cat17 = torch.cat([conv16, rb17], 1)
        conv17 = self.conv17(cat17)
        rb18 = self.rb18(conv17)
        cat18 = torch.cat([cat17, rb18], 1)
        conv18 = self.conv18(cat18)
        rb19 = self.rb19(conv18)
        cat19 = torch.cat([cat18, rb19], 1)
        conv19 = self.conv19(cat19)
        rb20 = self.rb20(conv19)
        cat20 = torch.cat([cat19, rb20], 1)
        conv20 = self.conv20(cat20)
        out6 = torch.add(conv20, out5)

        # Long skip connection(LSC) module
        out = torch.add(out6, out)
        out = self.upsampling(out)

        out = self.tail(out)

        out = out.div_(255.).add_(self.mean)
        out = torch.clamp_(out, 0.0, 1.0)

        return out
