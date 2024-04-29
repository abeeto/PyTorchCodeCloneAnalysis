from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

BATCH_SIZE = 32
IMAGE_SIZE = 28*28
HIDDEN_SIZE = 256
LATENT_SIZE = 64
EPOCHS = 30

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))
])


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0)).resize((500, 500)).show()


mnist_data = torchvision.datasets.MNIST('./mnist', train=True, transform=transform)
data_loader = Data.DataLoader(dataset=mnist_data, batch_size=BATCH_SIZE, shuffle=True)
TOTAL_STEPS = len(data_loader)
# print(next(iter(dataloader))[0][0][0].shape)
# show_image(next(iter(dataloader))[0][0][0])


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class G(nn.Module):
    def __init__(self, input_size, num_features):
        super(G, self).__init__()
        self.fc = nn.Linear(input_size, num_features)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsampel1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsampel1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


discriminator = D()
generator = G(LATENT_SIZE, 3136)
generator.load_state_dict(torch.load('GAN2_g'))
discriminator.load_state_dict(torch.load('GAN2_d'))
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)

# for epoch in range(EPOCHS):
#     for i, (images, _) in enumerate(data_loader):
#         batch_size = images.shape[0]
#         # images = images.reshape(batch_size, IMAGE_SIZE)
#
#         real_labels = torch.ones(batch_size, 1)
#         fake_labels = torch.zeros(batch_size, 1)
#
#         outputs = discriminator(images)
#         d_loss_real = loss_fn(outputs, real_labels)
#         real_score = outputs
#
#         z = torch.randn(batch_size, LATENT_SIZE)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images.detach())
#         d_loss_fake = loss_fn(outputs, fake_labels)
#         fake_score = outputs
#
#         # optimize discriminator
#         d_loss = d_loss_fake + d_loss_real
#         d_optimizer.zero_grad()
#         d_loss.backward()
#         d_optimizer.step()
#
#         # optimize generator
#         z = torch.randn(batch_size, LATENT_SIZE)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images)
#         g_loss = loss_fn(outputs, real_labels)
#
#         # d_optimizer.zero_grad()
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()
#
#         if i % 200 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
#                   .format(epoch, EPOCHS, i, TOTAL_STEPS, d_loss.item(), g_loss.item(), real_score.mean().item(),
#                           fake_score.mean().item()))
#             torch.save(discriminator.state_dict(), 'GAN2_d')
#             torch.save(generator.state_dict(), 'GAN2_g')
#
# torch.save(discriminator.state_dict(), 'GAN2_d')
# torch.save(generator.state_dict(), 'GAN2_g')

def show_image(image):
    transforms.ToPILImage()(image.squeeze(0)).resize((300, 300)).show()


generator.load_state_dict(torch.load('GAN2_g'))
discriminator.load_state_dict(torch.load('GAN2_d'))
show_image(next(iter(data_loader))[0][1][0])
# print(discriminator(next(iter(data_loader))[0][1][0].reshape(1,1,28,28)))
images = generator(torch.randn(32, LATENT_SIZE))
images = images.view(32, 28, 28)
# show_image(next(iter(data_loader))[0][0])
show_image(images[11])
