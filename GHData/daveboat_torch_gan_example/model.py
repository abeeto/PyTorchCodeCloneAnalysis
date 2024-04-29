import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    generator model. takes a 100 length array (random noise) as input, and does the following:
    FC to 7x7x256
    reshape to (7, 7, 256)
    conv2dtranspose (deconvolution) multiple times to get to (28, 28, 1)
    """
    def __init__(self, noise_dim=100):
        super().__init__()

        self.fc1 = nn.Linear(noise_dim, 7*7*256)
        self.batchnorm1 = nn.BatchNorm1d(7*7*256)
        self.leakyrelu1 = nn.LeakyReLU()

        self.conv2dtranspose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=(1, 1),
                                                  padding=2, bias=False, dilation=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU()

        self.conv2dtranspose3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                                  padding=2, bias=False, output_padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.leakyrelu3 = nn.LeakyReLU()

        self.conv2dtranspose4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5, 5), stride=(2, 2),
                                                  padding=2, bias=False, output_padding=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        # [N, 12544]
        x = x.view((-1, 256, 7, 7))
        # [N, 256, 7, 7]
        x = self.conv2dtranspose2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        # [N, 128, 7, 7]
        x = self.conv2dtranspose3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)
        # [N, 64, 14, 14]
        x = self.conv2dtranspose4(x)
        # [N, 1, 28, 28]
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):
    """
    The discriminator model is a model which tries to classify inputs as either real or generated, using a CNN
    We will use binary_cross_entropy_from_logits, so this module returns logits
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.leakyrelu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.leakyrelu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(128*7*7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)
        # [N, 64, 14, 14]
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.dropout2(x)
        # [N, 64, 7, 7]
        x = x.view((-1, 128*7*7))
        # [N, 128*7*7]
        x = self.fc(x)
        # [N, 1]

        return x

if __name__ == '__main__':
    x = torch.randn((10, 100))

    g = Generator()

    y = g(x)

    d = Discriminator()

    x = d(y)