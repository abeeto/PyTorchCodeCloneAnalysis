import torch
import torch.nn as nn

from generator import Generator

class Discriminator(nn.Module):
    def __init__(self, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(image_size, image_size * 2, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer5 = nn.Conv2d(image_size * 8, 1, kernel_size=4)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


if __name__ == '__main__':
    # for debug
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(image_size=64)

    input_rnd = torch.randn(1, 20)
    input_rnd = input_rnd.view(1, 20, 1, 1)
    output_image = G(input_rnd)
    diff = D(output_image)
