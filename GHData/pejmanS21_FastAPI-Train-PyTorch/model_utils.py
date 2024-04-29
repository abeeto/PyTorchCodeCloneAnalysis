import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv_block(in_channels, out_channels):
    m = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),
    )
    return m


class CNN(nn.Module):
    def __init__(
        self, in_channels: int = 1, hidden_size: int = 32, out_classes: int = 10
    ):
        super(CNN, self).__init__()
        self.out_classes = out_classes
        # in: (None, in_channels, 28, 28)
        self.conv_layer1 = conv_block(
            in_channels, hidden_size
        )  # (None, hidden_size, 14, 14)
        self.conv_layer2 = conv_block(
            hidden_size, hidden_size * 2
        )  # (None, hidden_size * 2, 7, 7)
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )  # (None, hidden_size * 2, 3, 3)

        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = nn.Linear(x.shape[1], 128)(x)
        x = self.drop(x)
        x = nn.Linear(128, self.out_classes)(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = CNN(1, 32, 10)
    summary(model, (1, 28, 28))
