import os
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.net(output)

        return output


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.net(x)
        img_shape = (1, 28, 28)
        output = output.view(output.shape[0], *img_shape)

        return output


transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5])
])


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(D, G, optimizer_D, optimizer_G, noise_dim):
    train_dataset = MNIST(root='data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(100):
        for i, (x, _) in enumerate(train_loader):
            num = x.shape[0]

            # train discriminator
            x_var = Variable(x.float().cuda())
            y_real = D(x_var)

            noise = torch.Tensor(np.random.normal(0, 1, (num, noise_dim)))
            noise_var = Variable(noise.cuda())
            fake_images = G(noise_var)
            y_fake = D(G(noise_var))
            gradient_penalty = compute_gradient_penalty(D, x_var.data, fake_images.data)

            loss_D = -torch.mean(y_real) + torch.mean(y_fake) + lambda_gp * gradient_penalty
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if i % 5 == 0 and i != 0:
                # train generator
                fake_images = G(noise_var)
                loss_G = -torch.mean(D(fake_images))
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                if i % 50 == 0:
                    print("Epoch: {:2d} | Iteration: {:<4d} | D_loss: {:.4f} | G_loss:{:.4f}".format(
                        epoch, i, loss_D.data.cpu().item(), loss_G.cpu().item()))

        fake_images = G(Variable(fix_noise).cuda())
        save_image(fake_images.data.cpu(), "WGAN_result/epoch%d.png" % epoch, nrow=5, normalize=True)


if __name__ == "__main__":
    if not os.path.exists('WGAN_result'):
        os.mkdir('WGAN_result')

    noise_dim = 100
    lambda_gp = 10
    D = Discriminator(input_size=784).cuda()
    G = Generator(noise_dim=noise_dim, output_dim=784).cuda()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    fix_noise = torch.Tensor(np.random.normal(0, 1, (25, noise_dim)))

    train(D, G, optimizer_D, optimizer_G, noise_dim)
