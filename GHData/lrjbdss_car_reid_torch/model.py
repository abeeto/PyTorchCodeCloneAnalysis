import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class EmbeddingNet(nn.Module):

    cfg = [64, 128, 256, 512]
    n_colors = 7
    n_models = 250

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        conv1 = nn.Conv2d(3, 32, 3, stride=3, padding=2, dilation=2)

        self.backbone = nn.Sequential(
            conv1, nn.BatchNorm2d(32), nn.PReLU(),
            self._make_layers(32),
            nn.AdaptiveAvgPool2d(1)
        )

        self.color_linear = nn.Linear(self.cfg[-1], self.n_colors)
        self.model_linear = nn.Linear(self.cfg[-1], self.n_models)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels in self.cfg:
            layers.append(Block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        features = x.view(x.size(0), -1)
        colors = F.softmax(self.color_linear(features), dim=1)
        models = F.softmax(self.model_linear(features), dim=1)
        return colors, models
