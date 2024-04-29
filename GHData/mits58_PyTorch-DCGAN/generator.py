import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        """
        z_dim : 入力乱数の次元
        image_size : 入力画像サイズ
        ------
        気持ちとしては、入力画像をガッと引き伸ばした後に、
        徐々に特徴を取り出していく感じ？
        """
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4),
                                    nn.BatchNorm2d(image_size * 8),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                                       kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(image_size * 4),
                                    nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                                       kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(image_size * 2),
                                    nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(nn.ConvTranspose2d(image_size * 2, image_size,
                                                       kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(image_size),
                                    nn.ReLU(inplace=True))

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())


    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


if __name__ == '__main__':
    # for debug
    import matplotlib.pyplot as plt

    G = Generator(z_dim=20, image_size=64)
    input_rnd = torch.randn(1, 20)
    input_rnd = input_rnd.view(1, 20, 1, 1)

    output_image = G(input_rnd)

    img = output_image[0][0].detach().numpy()
    plt.imshow(img, 'gray')
    plt.show()
