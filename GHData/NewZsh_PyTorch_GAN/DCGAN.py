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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5), (0.5))
])

data_train = torchvision.datasets.MNIST('/data_share/zhangsiheng/extra', transform=transform, train=True)

batch_size = 128
true_loader = DataLoader(dataset = data_train,
                        batch_size = batch_size,
                        shuffle = True,
                        num_workers = 4)

if not os.path.exists('/data_share/zhangsiheng/GAN/'):
    os.makedirs('/data_share/zhangsiheng/GAN/')
if not os.path.exists('/data_share/zhangsiheng/GAN/MNIST'):
    os.makedirs('/data_share/zhangsiheng/GAN/MNIST')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=128+10, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        validity = self.linear(torch.cat([x, c], -1))
        return validity

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        self.embedding = nn.Linear(in_features=10, out_features=input_size)
        self.linear = nn.Linear(in_features=input_size, out_features=4*4*256)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = self.embedding(c) + z
        x = self.linear(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.deconv(x)
        return x

noise_size = 100
D = Discriminator().float().to(device)
G = Generator(input_size=noise_size).float().to(device)
G.apply(weights_init_normal)
D.apply(weights_init_normal)

optimizer_d = optim.Adam(D.parameters(), lr=1e-4)
optimizer_g = optim.Adam(G.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

y_true = torch.ones((batch_size, 1)).float().to(device)
y_fake = torch.zeros((batch_size, 1)).float().to(device)

fixed_z = torch.randn([100, noise_size]).float().to(device)
fixed_c = np.zeros(100).astype(np.int64)
for i in range(100):
    fixed_c[i] = i % 10
fixed_c = torch.from_numpy(fixed_c)
fixed_c = F.one_hot(fixed_c, num_classes=len(data_train.classes)).float().to(device)

epochs = 200
steps = data_train.data.size(0) // batch_size

for epoch in range(1, epochs+1):
    G.train()
    with trange(steps) as t:
        for idx in t:
            real_imgs, real_labels = next(iter(true_loader))
            real_imgs = real_imgs.to(device)
            real_labels = F.one_hot(real_labels, num_classes = len(data_train.classes)).float().to(device)
            outputs = D(real_imgs, real_labels)
            d_loss1 = criterion(outputs, y_true)
            
            noise = torch.randn((batch_size, noise_size)).float().to(device)
            fake_labels = torch.from_numpy(np.random.choice(len(data_train.classes), batch_size)).view(batch_size)
            fake_labels = F.one_hot(fake_labels, num_classes=len(data_train.classes)).float().to(device)
            fake_imgs = G(noise, fake_labels)
            outputs = D(fake_imgs.detach(), fake_labels)
            d_loss2 = criterion(outputs, y_fake)

            optimizer_d.zero_grad()
            d_loss = d_loss1 + d_loss2
            d_loss.backward()
            optimizer_d.step()

            outputs = D(fake_imgs, fake_labels)
            g_loss = criterion(outputs, y_true)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            t.set_postfix(train_epoch=epoch, idx=idx, d_loss=d_loss.item(), g_loss=g_loss.item())

    if epoch % 10 == 0:
        G.eval()
        fixed_fake_images = G(fixed_z, fixed_c)
        save_image(fixed_fake_images, '/data_share/zhangsiheng/GAN/MNIST/DCGAN_{}.png'.format(epoch), nrow=10, normalize=True)

        torch.save(G.state_dict(), '/data_share/zhangsiheng/GAN/MNIST/DCGAN.pth')