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
import torchvision.datasets as datasets
from torch import autograd


BATCH_SIZE = 64
IMAGE_SIZE = 3 * 128 * 128
HIDDEN_SIZE = 256
LATENT_SIZE = 64
EPOCHS = 30

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])


data_set = datasets.ImageFolder(root=r'D:\BaiduNetdiskDownload\faces',
                                transform=transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
                                )

data_loader = Data.DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)
TOTAL_STEPS = len(data_loader)
# print(next(iter(dataloader))[0][0][0].shape)
# show_image(next(iter(dataloader))[0][0][0])


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(65536, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            # nn.Sigmoid()
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
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.downsampel1 = nn.Sequential(
            nn.Conv2d(3, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 128, 128)
        x = self.br(x)
        x = self.downsampel1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0)).resize((128, 128)).show()


def test_my_model(gen):
    z = torch.randn(5, LATENT_SIZE)
    ims = gen(z)
    ims = ims.view(5, 3, 128, 128)
    gen.eval()
    with torch.no_grad():
        for im in ims:
            show_image(im)
    gen.train()


def gradient_penalty(discriminator, real_images, fake_images):
    t = torch.rand(real_images.size(0), 1, 1, 1)
    t = t.expand_as(real_images)
    mid = t * real_images + (1-t) * fake_images
    mid.requires_grad_()
    pred = discriminator(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True,
                          only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp

discriminator = D()
generator = G(LATENT_SIZE, 128*128*3)
generator.load_state_dict(torch.load('MyGAN2_g'))
discriminator.load_state_dict(torch.load('MyGAN2_d'))

test_my_model(generator)
exit()

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.shape[0]
        images = images.reshape(batch_size, 3, 128, 128)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = discriminator(images)
        # d_loss_real = loss_fn(outputs, real_labels)
        d_loss_real = -outputs.mean()
        real_score = outputs

        z = torch.randn(batch_size, LATENT_SIZE)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        # d_loss_fake = loss_fn(outputs, fake_labels)
        d_loss_fake = outputs.mean()
        fake_score = outputs

        gp = gradient_penalty(discriminator, images, fake_images)
        # optimize discriminator
        d_loss = d_loss_fake + d_loss_real + 0.2*gp
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # optimize generator
        z = torch.randn(batch_size, LATENT_SIZE)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        # g_loss = loss_fn(outputs, real_labels)
        g_loss = -outputs.mean()

        # d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 25 == 0 and i != 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, EPOCHS, i, TOTAL_STEPS, d_loss.item(), g_loss.item(), real_score.mean().item(),
                          fake_score.mean().item()))
            torch.save(discriminator.state_dict(), 'MyGAN2_d')
            torch.save(generator.state_dict(), 'MyGAN2_g')


torch.save(discriminator.state_dict(), 'MyGAN2_d')
torch.save(generator.state_dict(), 'MyGAN2_g')





# generator.load_state_dict(torch.load('MyGAN2_g'))
# discriminator.load_state_dict(torch.load('MyGAN2_d'))
# show_image(next(iter(data_loader))[0][1])
# # print(discriminator(next(iter(data_loader))[0][1][0].reshape(1,1,28,28)))
# images = generator(torch.randn(BATCH_SIZE, LATENT_SIZE))
# images = images.view(BATCH_SIZE, 3, 128, 128)
# # show_image(next(iter(data_loader))[0][0])
# show_image(images[15])
