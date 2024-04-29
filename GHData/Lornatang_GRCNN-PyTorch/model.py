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
    "GRCNN",
]


class _BidirectionalLSTM(nn.Module):

    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super(_BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        sequence_length, batch_size, inputs_size = recurrent.size()
        sequence_length2 = recurrent.view(sequence_length * batch_size, inputs_size)

        out = self.linear(sequence_length2)  # [sequence_length * batch_size, output_size]
        out = out.view(sequence_length, batch_size, -1)  # [sequence_length, batch_size, output_size]

        return out


class _GRCLUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_GRCLUnit, self).__init__()
        # See `https://dl.acm.org/doi/pdf/10.5555/3294771.3294803` formula (3)
        self.wgf_bn = nn.BatchNorm2d(channels)
        self.wgr_bn = nn.BatchNorm2d(channels)
        self.wf_bn = nn.BatchNorm2d(channels)
        self.wx_bn = nn.BatchNorm2d(channels)
        self.gx_bn = nn.BatchNorm2d(channels)

    def forward(self, wgf: torch.Tensor, wgx: torch.Tensor, wf: torch.Tensor, wx: torch.Tensor):
        gated = torch.sigmoid(self.wgf_bn(wgf) + self.wgr_bn(wgx))
        wf = self.wf_bn(wf)
        wx = self.wx_bn(wx)

        out = torch.relu(wf + self.gx_bn(wx * gated))

        return out


class _GRCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_iterations: int) -> None:
        super(_GRCL, self).__init__()
        self.num_iterations = num_iterations
        # See `https://dl.acm.org/doi/pdf/10.5555/3294771.3294803` formula (4)
        self.wgf_u = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.wgr_x = nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.wf_u = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.wg_x = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_x = nn.BatchNorm2d(out_channels)

        self.GRCL = [_GRCLUnit(out_channels) for _ in range(num_iterations)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, inputs):
        wgf = self.wgf_u(inputs)
        wf = self.wf_u(inputs)
        out = torch.relu(self.bn_x(wf))

        for i in range(self.num_iterations):
            out = self.GRCL[i](wgf, self.wgr_x(out), wf, self.wg_x(out))

        return out


class GRCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(GRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),

            _GRCL(64, 64, 5),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),

            _GRCL(64, 128, 5),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            _GRCL(128, 256, 5),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(256, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.recurrent = nn.Sequential(
            _BidirectionalLSTM(512, 256, 256),
            _BidirectionalLSTM(256, 256, num_classes),

        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Feature sequence
        features = self.features(x)  # [b, c, h, w]
        features = features.squeeze(2)  # [b, c, w]
        features = features.permute(2, 0, 1)  # [w, b, c]

        # Deep bidirectional LSTM
        out = self.recurrent(features)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
