

import torch
print(torch.cuda.is_available(), torch.__version__)


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision


noise_dim = 100
batch_size = 32
num_epochs = 20



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256*7*7, bias=False)
        self.bn1 = nn.BatchNorm1d(256*7*7)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, \
                                              kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, \
                                                  stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=5, \
                                                  stride=2, padding=2, output_padding=1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = torch.reshape(x, (-1,256,7,7))

        x = self.conv_transpose1(x)
        x = F.leaky_relu(self.bn2(x))

        x = self.conv_transpose2(x)
        x = F.leaky_relu(self.bn3(x))

        x = self.conv_transpose3(x)
        return x




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.drop = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(7*7*128, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = torch.reshape(x, (-1,7*7*128))
        x = torch.sigmoid(self.fc(x))
        return x


def train(generator, discriminator, data_loader, optimizer_disc, optimizer_gen, device, epoch, writer):
    num_batches = len(data_loader.dataset)//batch_size
    for idx, (real_images, _) in enumerate(data_loader):
        real_images = real_images.to(device)
        generator.train()
        discriminator.train()
        noise = torch.randn(batch_size, noise_dim)
        noise = noise.to(device)
        fake_images = generator(noise)

        probs_real = discriminator(real_images)
        probs_fake = discriminator(fake_images)
        ones_tensor = torch.ones_like(probs_real)
        zeros_tensor = torch.zeros_like(probs_fake)
        disc_loss_real = F.binary_cross_entropy(input=probs_real, target=ones_tensor)
        disc_loss_fake = F.binary_cross_entropy(input=probs_fake, target=zeros_tensor)
        disc_loss = 0.5*(disc_loss_real + disc_loss_fake)
        disc_loss.backward(retain_graph=True)
        optimizer_disc.step()

        generator.train()
        discriminator.eval()
        discriminator.eval()
        gen_loss = F.binary_cross_entropy(input=probs_fake, target=ones_tensor)
        gen_loss.backward()
        optimizer_gen.step()

        print("[Train] Epoch: {} [{}/{}]    Generator Loss: {:.6f}   Discriminator Loss: {:.6f}".format(
              epoch, idx*batch_size, len(data_loader.dataset),
              disc_loss.item(), gen_loss.item()))
        writer.add_scalar('Loss/Generator', gen_loss.item(), num_batches*epoch+idx)
        writer.add_scalar('Loss/Discriminator_Real', disc_loss_real.item(), num_batches*epoch+idx)
        writer.add_scalar('Loss/Discriminator_Fake', disc_loss_fake.item(), num_batches*epoch+idx)



def test(generator, data_loader, device, epoch, writer):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(batch_size, noise_dim)
        noise = noise.to(device)
        fake_images = generator(noise)

        grid = torchvision.utils.make_grid(fake_images)
        writer.add_image('images', grid, 0)
        writer.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator()
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    transform = transforms.Compose([transforms.ToTensor()]) # converts to 0-1 and makes it (M, 1, H, W)
    dataset = datasets.MNIST('/tmp/', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    def concat(a, b):
        yield from a
        yield from b

    optimizer_disc = optim.Adam(params=discriminator.parameters(), lr=1e-3)
    optimizer_gen = optim.Adam(params=generator.parameters(), lr=1e-3)

    writer = SummaryWriter()
    for epoch in range(num_epochs):
        train(generator, discriminator, data_loader, optimizer_disc, optimizer_gen, device, epoch, writer)
        test(generator, data_loader, device, epoch, writer)



if __name__=="__main__":
    main()
