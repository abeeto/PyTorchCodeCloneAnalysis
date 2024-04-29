from torch import nn


class SeNet(nn.Module):
    def __init__(self, channel, reduction):
        super(SeNet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel//reduction),
            nn.ReLU(),
            nn.Linear(in_features=channel//reduction, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        bn, c, _, _ = x.shape()
        y = self.squeeze(x).view(bn, c,)
        y = self.excitation(y).view(bn, c, 1, 1)
        out = x*y.expand_as(x)
        return out
