import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from torchvision import transforms
import torch.utils.data as Data
import torchvision.datasets as datasets

LATENT_SPACE = 2
HIDDEN_SIZE = 400
G_OUTPUT_SIZE = IMAGE_SIZE = 3*128*128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 5e-4


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_SPACE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, G_OUTPUT_SIZE)
        )

    def forward(self, z):
        output = self.net(z)
        return output
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(G_OUTPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(True),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self.net(z)
        return output


def get_dataset(path=r'D:\BaiduNetdiskDownload\faces'):
    data_set = datasets.ImageFolder(root=path,
                                    transform=transforms.Compose([
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor()])
                                    )
    return data_set


def get_dataloader(dataset=get_dataset()):
    data_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0).reshape(3, 128, 128)).show()


def gradient_penalty(discriminator, real_images, fake_images):
    t = torch.rand(BATCH_SIZE, 1)
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


def main():
    generator = Generator()
    discriminator = Discriminator()
    data_set = get_dataset()
    data_loader = get_dataloader(data_set)
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    for epoch in range(EPOCHS):
        # train discriminator
        for i, (images, _) in enumerate(data_loader):
            # train on real images
            images = images.reshape(BATCH_SIZE, IMAGE_SIZE)
            # print(images.shape)
            pred_real = discriminator(images)
            loss_real = -pred_real.mean()

            # train on fake images
            z = torch.randn(BATCH_SIZE, LATENT_SPACE)
            # detach作用：停止继续往前传梯度，减少计算量
            fake_images = generator(z).detach()
            pred_fake = discriminator(fake_images)
            loss_fake = pred_fake.mean()

            gp = gradient_penalty(discriminator, images, fake_images)
            loss_d = loss_fake + loss_real + 0.2*gp

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # train generator
            z = torch.randn(BATCH_SIZE, LATENT_SPACE)
            fake_images = generator(z)
            pred_fake = discriminator(fake_images)

            loss_g = -pred_fake.mean()

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if i % 200 == 0:
                print(loss_d.item(), loss_g.item())
                show_image(fake_images[0])


if __name__ == '__main__':
    main()
