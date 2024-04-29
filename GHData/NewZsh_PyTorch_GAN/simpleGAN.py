import torch 
import torchvision
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import transforms
from torchvision.utils import save_image

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import trange

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5), (0.5))
])

data_train = torchvision.datasets.MNIST('/data_share/zhangsiheng/extra', train=True)
x_train = transform(data_train.data)
y_train = F.one_hot(data_train.targets, num_classes=len(data_train.classes))

data_test = torchvision.datasets.MNIST('/data_share/zhangsiheng/extra', train=False)
x_test = transform(data_test.data)
y_test = F.one_hot(data_test.targets, num_classes=len(data_train.classes))

x_train = x_train.view(-1, 1, 28, 28)
x_test = x_test.view(-1, 1, 28, 28)

true_loader = DataLoader(dataset = TensorDataset(x_train),
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 4)

if not os.path.exists('/data_share/zhangsiheng/GAN/'):
    os.makedirs('/data_share/zhangsiheng/GAN/')
if not os.path.exists('/data_share/zhangsiheng/GAN/MNIST'):
    os.makedirs('/data_share/zhangsiheng/GAN/MNIST')

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Linear(in_features=28*28, out_features=512, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        layers= []
        layers.append(nn.Linear(in_features=input_size, out_features=128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=128, out_features=256))
        layers.append(nn.BatchNorm1d(256, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=512))
        layers.append(nn.BatchNorm1d(512, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=28*28))
        layers.append(nn.Tanh()) #[-1,1]

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 1, 28, 28)
        return x

noise_size = 100
D = Discriminator().float().to(device)
G = Generator(input_size=noise_size).float().to(device)

optimizer_d = optim.Adam(D.parameters(), lr=1e-4)
optimizer_g = optim.Adam(G.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

batch_size = 32
y_true = torch.ones((batch_size, 1)).float().to(device)
y_fake = torch.zeros((batch_size, 1)).float().to(device)

fixed_z = torch.randn([100, noise_size]).float().to(device)

epochs = 200
steps = x_train.size(0) // batch_size

for epoch in range(1, epochs+1):
    G.train()
    with trange(steps) as t:
        for idx in t:
            inputs = next(iter(true_loader))[0].to(device)
            outputs = D(inputs)
            d_loss1 = criterion(outputs, y_true)

            noise = torch.randn((batch_size, noise_size)).float().to(device)
            fake = G(noise)
            outputs = D(fake.detach())
            d_loss2 = criterion(outputs, y_fake)

            optimizer_d.zero_grad()
            d_loss = d_loss1 + d_loss2
            d_loss.backward()
            optimizer_d.step()

            outputs = D(fake)
            g_loss = criterion(outputs, y_true)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            t.set_postfix(train_epoch=epoch, idx=idx, d_loss=d_loss.item(), g_loss=g_loss.item())
    
    if epoch % 10 == 0:
        G.eval()
        fixed_fake_images = G(fixed_z)
        save_image(fixed_fake_images, '/data_share/zhangsiheng/GAN/MNIST/SimpleGAN_{}.png'.format(epoch), nrow=10, normalize=True)

        torch.save(G.state_dict(), '/data_share/zhangsiheng/GAN/MNIST/simpleGAN.pth')
