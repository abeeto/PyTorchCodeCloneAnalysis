import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    PyTorch implementation of VGG16 model.
    Paper: https://arxiv.org/abs/1409.1556
    """
    def __init__(self, num_classes: int = 1000, in_channels: int = 3, dropout_rate: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self.create_features()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def create_features(self):
        in_channels = self.in_channels
        layers = list()
        for layer in self.cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=(3, 3), padding=(1, 1)),
                           nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)


def main():
    model = VGG16()
    x = torch.randn(50, 3, 224, 224)
    model.eval()
    output = model(x)
    assert output.shape == (50, 1000)


if __name__ == '__main__':
    main()

