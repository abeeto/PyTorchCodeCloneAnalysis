import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

import matplotlib.gridspec as gridspec
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


def loadMNIST(batch_size):
    trains_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST("./data", train=True, transform=trains_img, download=True)
    testset = MNIST("./data", train=False, transform=trains_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset, testset, trainloader, testloader

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )

        self.fc=nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),

            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x

if __name__=="__main__":
    criterion = nn.BCELoss()
    num_img = 100
    z_dimension = 100
    D = discriminator().cuda()
    G = generator(z_dimension, 3136).cuda()
    trainset, testset, trainloader, testloader = loadMNIST(num_img)
    d_optimizer = optim.Adam(D.parameters(), lr = 0.0003)
    g_optimizer = optim.Adam(G.parameters(), lr = 0.0003)

    count = 0
    epoch = 100
    gepoch = 1
    print("start")
    for i in range(epoch):
        for (img, label) in trainloader:
            img = Variable(img).cuda()
            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()

            real_out = D(img)
            d_loss_real = criterion(real_out, real_label)
            real_score =real_out

            z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(z)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_score = fake_out

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            for j in range(gepoch):
                fake_label = Variable(torch.ones(num_img)).cuda()
                z = Variable(torch.randn(num_img, z_dimension)).cuda()
                fake_img = G(z)
                output = D(fake_img)
                g_loss = criterion(output, fake_label)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(
            i, epoch, d_loss.data, g_loss.data,
            real_score.data.mean(), fake_score.data.mean()))
        
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './dc_img/fake_images-{}.png'.format(i+1))
#        plt.show()
#        count = count + 1

