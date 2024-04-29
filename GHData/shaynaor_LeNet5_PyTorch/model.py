import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    PyTorch implementation of LeNet5 model. Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    Note: Using regular Thanh, AvgPool2d and SoftMax for simplifications.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    model = LeNet5(10)
    x = torch.randn(1, 1, 32, 32)
    model.eval()
    output = model(x)
    assert output.shape == (1, 10)


if __name__ == '__main__':
    main()

