import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim
import numpy as np


transform = transforms.Compose([transforms.Scale(32),
                                transforms.ToTensor(),
                                ])


train_set = torchvision.datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=40,
                                           shuffle=True,
                                           num_workers=2)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )
        # print(self.net)
        #input()

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=7, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        input(x)
        # print('x ', x.size())
        x = x.view(x.size(0), 128, 1, 1)
        out = self.net(x)
        return out


D = Discriminator()
G = Generator()
D.cuda()
G.cuda()

criterion = nn.BCELoss()

G_optimizer = torch.optim.Adam(G.parameters(), lr=0.002)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.002)

for epoch in range(50):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())

        D.zero_grad()
        # print(images.size())
        outputs = D(images)
        real_loss = criterion(outputs, real_labels)
        real_score = outputs

        noise = Variable(torch.randn(images.size(0), 128)).cuda()

        fake_images = G(noise)
        # input(fake_images.size())
        # input(fake_labels.size())
        outputs = D(fake_images.detach())  # Returns a new Variable, detached from the current graph.
        fake_loss = criterion(outputs, fake_labels)
        fake_score = outputs

        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()

        G.zero_grad()
        noise = Variable(torch.randn(images.size(0), 128)).cuda()
        fake_images = G(noise)
        outputs = D(fake_images)
        G_loss = criterion(outputs, real_labels)
        G_loss.backward()
        G_optimizer.step()

        if (i + 1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 50, i + 1, 600, D_loss.data[0], G_loss.data[0],
                     real_score.data.mean(), fake_score.cpu().data.mean()))

    fake_images = fake_images.view(fake_images.size(0), 1, 32, 32)
    torchvision.utils.save_image(fake_images.data,
                                 './data/fake_samples_%d.png' % (epoch))

torch.save(Generator.state_dict(), './generator.pkl')
torch.save(Discriminator.state_dict(), './discriminator.pkl')



