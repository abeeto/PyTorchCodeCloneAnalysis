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
from functools import partial
from typing import Any, List, Union, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth

__all__ = [
    "ConvNeXt",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
]

convnext_tiny_cfg = [
    [96, 192, 3],
    [192, 384, 3],
    [384, 768, 9],
    [768, None, 3],
]

convnext_small_cfg = [
    [96, 192, 3],
    [192, 384, 3],
    [384, 768, 27],
    [768, None, 3],
]

convnext_base_cfg = [
    [128, 256, 3],
    [256, 512, 3],
    [512, 1024, 27],
    [1024, None, 3],
]

convnext_large_cfg = [
    [192, 384, 3],
    [384, 768, 3],
    [768, 1536, 27],
    [1536, None, 3],
]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        out = x.permute(0, 2, 3, 1)
        out = F.layer_norm(out, self.normalized_shape, self.weight, self.bias, self.eps)
        out = out.permute(0, 3, 1, 2)

        return out


class CNBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            layer_scale: float,
            stochastic_depth_prob: float,
    ) -> None:
        super(CNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 7), (1, 1), (3, 3), groups=channels, bias=True),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.block(x)
        out = torch.mul(out, self.layer_scale)
        out = self.stochastic_depth(out)
        out = torch.add(out, identity)

        return out


class ConvNeXt(nn.Module):

    def __init__(
            self,
            arch_cfg: [Union[int, Optional[int], int]],
            stochastic_depth_prob: float = 0.2,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
    ) -> None:
        super(ConvNeXt, self).__init__()
        features: List[nn.Module] = [Conv2dNormActivation(
            3,
            arch_cfg[0][0],
            kernel_size=4,
            stride=4,
            padding=0,
            groups=1,
            norm_layer=partial(LayerNorm2d, eps=1e-6),
            activation_layer=None,
            bias=True,
        )]

        total_stage_blocks = sum(cfg[2] for cfg in arch_cfg)
        stage_block_id = 0
        for cfg in arch_cfg:
            stage: List[nn.Module] = []
            for _ in range(cfg[2]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(CNBlock(cfg[0], layer_scale, sd_prob))
                stage_block_id += 1

            features.append(nn.Sequential(*stage))
            if cfg[1] is not None:
                # Downsampling
                features.append(
                    nn.Sequential(
                        LayerNorm2d(cfg[0]),
                        nn.Conv2d(cfg[0], cfg[1], (2, 2), (2, 2), (0, 0)),
                    )
                )

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        last_channels = arch_cfg[-1][1] if arch_cfg[-1][1] is not None else arch_cfg[-1][0]
        self.classifier = nn.Sequential(
            LayerNorm2d(last_channels),
            nn.Flatten(1),
            nn.Linear(last_channels, num_classes)
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def convnext_tiny(**kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(convnext_tiny_cfg, 0.1, **kwargs)

    return model


def convnext_small(**kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(convnext_small_cfg, 0.4, **kwargs)

    return model


def convnext_base(**kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(convnext_base_cfg, 0.5, **kwargs)

    return model


def convnext_large(**kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(convnext_large_cfg, 0.5, **kwargs)

    return model
