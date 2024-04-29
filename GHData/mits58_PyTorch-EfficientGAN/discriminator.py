import torch
import torch.nn as nn

from generator import Generator


class Discriminator(nn.Module):
    def __init__(self, z_dim=20):
        super(Discriminator, self).__init__()

        # --- 入力画像に対する層 --- #
        self.img_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.img_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # --- 入力乱数に対する層 --- #
        self.rnd_layer1 = nn.Linear(z_dim, 512)

        # --- 判定層 --- #
        self.layer1 = nn.Sequential(
            nn.Linear(3648, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Linear(1024, 1)

    def forward(self, img, z):
        img_out = self.img_layer1(img)
        img_out = self.img_layer2(img_out)

        z = z.view(z.shape[0], -1)
        z_out = self.rnd_layer1(z)

        img_out = img_out.view(-1, 64 * 7 * 7)
        out = torch.cat([img_out, z_out], dim=1)
        out = self.layer1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        out = self.layer2(out)

        return out, feature


if __name__ == '__main__':
    G = Generator(z_dim=20)
    D = Discriminator(z_dim=20)

    input_z = torch.randn(2, 20)
    fake_images = G(input_z)

    d_out, _ = D(fake_images, input_z)

    print(nn.Sigmoid()(d_out))
