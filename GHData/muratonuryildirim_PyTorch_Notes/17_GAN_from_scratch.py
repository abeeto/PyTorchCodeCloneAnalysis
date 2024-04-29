import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(nn.Linear(img_dim, 128),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(128, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(z_dim, 256),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(256, img_dim),
                                 nn.Tanh())  # to make all pixel between [-1,1]

    def forward(self, x):
        return self.gen(x)


# GANs are sensitive to Hyperparameters
# It is hard to get a good balance between disc and gen (disc should not be too harsh or too naive)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
num_epochs=10
lr = 3e-4
z_dim = 64
img_dim = 1 * 28 * 28  # MNIST

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307), (0.3081))])

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

dataset = torchvision.datasets.MNIST(root='dataset/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # generate image from noise
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        # train dicriminator
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # train generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

