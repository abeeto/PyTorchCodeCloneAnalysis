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

BATCH_SIZE = 32
IMAGE_SIZE = 3*128*128
HIDDEN_SIZE = 256
LATENT_SIZE = 64
EPOCHS = 30

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0).reshape(3, 128, 128)).show()


data_set = datasets.ImageFolder(root=r'D:\BaiduNetdiskDownload\faces',
                                transform=transforms.Compose([
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
                                )

data_loader = Data.DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)
TOTAL_STEPS = len(data_loader)

discriminator = nn.Sequential(
    nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
    nn.LeakyReLU(0.2),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.LeakyReLU(0.2),
    nn.Linear(HIDDEN_SIZE, 1),
    nn.Sigmoid()
)

generator = nn.Sequential(
    nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
    nn.ReLU(True),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(True),
    nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
    nn.Tanh()
)

def gradient_penalty(discriminator, real_images, fake_images):
    t = torch.rand(real_images.size(0), 1)
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

generator.load_state_dict(torch.load('MyGAN1_g'))
discriminator.load_state_dict(torch.load('MyGAN1_d'))
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.shape[0]
        images = images.reshape(batch_size, IMAGE_SIZE)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = discriminator(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, LATENT_SIZE)
        fake_images = generator(z).detach()
        outputs = discriminator(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        gp = gradient_penalty(discriminator, images, fake_images)
        # optimize discriminator
        d_loss = d_loss_fake + d_loss_real + gp
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # optimize generator
        z = torch.randn(batch_size, LATENT_SIZE)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = loss_fn(outputs, real_labels)

        # d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 200 == 0 and i != 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, EPOCHS, i, TOTAL_STEPS, d_loss.item(), g_loss.item(), real_score.mean().item(),
                          fake_score.mean().item()))
            torch.save(discriminator.state_dict(), 'MyGAN1_d')
            torch.save(generator.state_dict(), 'MyGAN1_g')
            show_image(fake_images[0])

torch.save(discriminator.state_dict(), 'MyGAN1_d')
torch.save(generator.state_dict(), 'MyGAN1_g')


# show_image(data_loader.dataset[30][0])

# generator.load_state_dict(torch.load('MyGAN1_g'))
# discriminator.load_state_dict(torch.load('MyGAN1_d'))
#
# images = generator(torch.randn(BATCH_SIZE, LATENT_SIZE))
# images = images.view(BATCH_SIZE, 3, 128, 128)
# print(images.shape)
# show_image(images[3])
