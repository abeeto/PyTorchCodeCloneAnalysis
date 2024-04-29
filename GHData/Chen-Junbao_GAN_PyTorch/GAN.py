import torch
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)

        return output


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.net(x)

        return output


transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5])
])


def deprocess(x):
    return (x + 1.0) / 2.0


def discriminator_loss(y_real, y_fake):
    num = y_real.shape[0]
    criterion = nn.BCELoss()
    real_loss = criterion(y_real, torch.ones(num, 1).cuda())
    fake_loss = criterion(y_fake, torch.zeros(num, 1).cuda())

    return real_loss + fake_loss


def generator_loss(y_fake):
    num = y_fake.shape[0]
    criterion = nn.BCELoss()
    loss = criterion(y_fake, torch.ones(num, 1).cuda())

    return loss


def train(D, G, D_optimizer, G_optimizer, noise_dim):
    figure, axis = plt.subplots(4, 4)
    plt.ion()

    train_dataset = MNIST(root='data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(100):
        for i, (x, _) in enumerate(train_loader):
            num = x.shape[0]

            # train discriminator
            x = torch.reshape(x, (-1, 784))
            x_var = Variable(x.cuda())
            y_real = D(x_var)

            noise = torch.randn((num, noise_dim))
            noise_var = Variable(noise.cuda())
            y_fake = D(G(noise_var))

            D_loss = discriminator_loss(y_real, y_fake)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # train generator
            noise = torch.randn((num, noise_dim))
            noise_var = Variable(noise.cuda())
            G_output = G(noise_var)
            y_fake = D(G_output)
            G_loss = generator_loss(y_fake)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            if i % 50 == 0:
                print("Epoch: {:2d} | Iteration: {:<4d} | D_loss: {:.4f} | G_loss:{:.4f}".format(
                    epoch, i, D_loss.data.cpu().item(), G_loss.cpu().item()))

                images = deprocess(G_output.data.cpu().numpy())
                for j in range(16):
                    axis[j // 4][j % 4].imshow(np.reshape(images[j], (28, 28)), cmap='gray')
                    axis[j // 4][j % 4].set_xticks(())
                    axis[j // 4][j % 4].set_yticks(())

                plt.suptitle("epoch: {} iteration: {}".format(epoch, i))
                plt.pause(0.01)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    noise_dim = 100
    D = Discriminator(input_size=784).cuda()
    G = Generator(noise_dim=noise_dim, output_dim=784).cuda()
    D_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    train(D, G, D_optimizer, G_optimizer, noise_dim)
