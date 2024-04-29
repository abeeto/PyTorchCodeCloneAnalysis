import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        out = x + out
        return out


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 2, 1),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)
