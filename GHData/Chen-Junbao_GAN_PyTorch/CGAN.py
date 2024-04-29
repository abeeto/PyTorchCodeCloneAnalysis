import os
import torch
import torch.nn as nn
import torchvision.transforms as tfs

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, d=128):
        super().__init__()

        self.deconv_label = nn.Sequential(
            nn.ConvTranspose2d(10, d * 2, 4, 1, 0),
            nn.BatchNorm2d(d * 2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(100, d * 2, 4, 1, 0),
            nn.BatchNorm2d(d * 2),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, y):
        x = self.deconv_1(x)
        y = self.deconv_label(y)
        x = torch.cat([x, y], 1)
        output = self.net(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, d=128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, d // 2, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_label = nn.Sequential(
            nn.Conv2d(10, d // 2, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv_label(y)
        x = torch.cat([x, y], 1)
        output = self.net(x)

        return output


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def train():
    train_loader = DataLoader(
        MNIST('data', train=True, download=False, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        for x, y in train_loader:
            num = x.shape[0]

            # train D
            label_D = labels_D[y]
            x_var = Variable(x.cuda())
            label_D_var = Variable(label_D.cuda())

            y_real = D(x_var, label_D_var).squeeze()
            loss_D_real = criterion(y_real, torch.ones(num).cuda())

            noise = torch.randn((num, 100)).reshape(-1, 100, 1, 1)
            noise_y = (torch.rand(num, 1) * 10).long().squeeze()
            label_G = labels_G[noise_y]
            label_D = labels_D[noise_y]
            noise_var = Variable(noise.cuda())
            label_G_var = Variable(label_G.cuda())
            label_D_var = Variable(label_D.cuda())

            fake_images = G(noise_var, label_G_var)
            y_fake = D(fake_images, label_D_var).squeeze()

            loss_D_fake = criterion(y_fake, torch.zeros(num).cuda())

            loss_D = loss_D_real + loss_D_fake

            D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # train G
            noise = torch.randn((num, 100)).reshape(-1, 100, 1, 1)
            noise_y = (torch.rand(num, 1) * 10).type(torch.LongTensor).squeeze()
            label_G = labels_G[noise_y]
            label_D = labels_D[noise_y]
            noise_var = Variable(noise.cuda())
            label_G_var = Variable(label_G.cuda())
            label_D_var = Variable(label_D.cuda())

            fake_images = G(noise_var, label_G_var)
            y_fake = D(fake_images, label_D_var).squeeze()

            loss_G = criterion(y_fake, torch.ones(num).cuda())

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            print("[Epoch: %d] [loss_D: %f] [loss_G: %f]" % (epoch, loss_D, loss_G))

        with torch.no_grad():
            fixed_fake_images = G(fixed_noise_var, fixed_y_label_var)
            save_image(fixed_fake_images, "WGAN_result/epoch" + str(epoch) + ".jpg", nrow=10, normalize=True)


if __name__ == "__main__":
    if not os.path.exists("WGAN_result"):
        os.mkdir("WGAN_result")

    batch_size = 128
    lr = 1e-4
    epochs = 20

    G = Generator().cuda()
    D = Discriminator().cuda()
    G.weight_init(0.0, 0.02)
    D.weight_init(0.0, 0.02)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # the labels used in G (1 * 10 * 1 * 1)
    labels_G = torch.zeros(10, 10)
    labels_G = labels_G.scatter_(1, torch.LongTensor(list(range(10))).reshape(10, 1), 1).reshape(10, 10, 1, 1)
    # the labels used in D (1 * 10 * 32 * 32)
    labels_D = torch.zeros([10, 10, 32, 32])
    for i in range(10):
        labels_D[i, i, :, :] = 1

    transform = tfs.Compose([
        tfs.Resize(32),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5], std=[0.5])
    ])

    # 100 fixed samples
    fixed_noise = torch.randn(100, 100).reshape(-1, 100, 1, 1)

    fixed_y = []
    for i in range(10):
        fixed_y += [i] * 10
    fixed_y = torch.LongTensor(fixed_y).reshape(-1, 1)
    fixed_label_G = labels_G[fixed_y].reshape(-1, 10, 1, 1)

    fixed_noise_var = Variable(fixed_noise.cuda())
    fixed_y_label_var = Variable(fixed_label_G.cuda())

    train()
