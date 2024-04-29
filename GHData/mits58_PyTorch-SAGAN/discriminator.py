import torch.nn as nn

from selfattention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size, image_size * 2,
                                             kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size * 2, image_size * 4,
                                             kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.selfattention1 = SelfAttention(input_dim=image_size * 4)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size * 4, image_size * 8,
                                             kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.selfattention2 = SelfAttention(input_dim=image_size * 8)

        self.layer5 = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, at_map1 = self.selfattention1(out)
        out = self.layer4(out)
        out, at_map2 = self.selfattention2(out)
        out = self.layer5(out)

        return out, at_map1, at_map2
