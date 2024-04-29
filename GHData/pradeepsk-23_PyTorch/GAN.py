import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
root_dir = "D:\IRP\GitHub\Frame"

stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
dataset = ImageFolder(root_dir, tt.Compose([tt.Resize(64), 
                                            tt.CenterCrop(64), 
                                            tt.ToTensor(),
                                            tt.Normalize(*stats)]))

# DataLoader (input pipeline)
batch_size = 128
train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)

# Discriminator
D = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True), # out: 64 x 32 x 32
                
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True), # out: 128 x 16 x 16
                
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True), # out: 256 x 8 x 8
                
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True), # out: 512 x 4 x 4
                
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False), # out: 1 x 1 x 1
                nn.Flatten(), 
                nn.Sigmoid())

# Generator
latent_size = 128*1*1
G = nn.Sequential(nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False), # in: 128x1x1 
                nn.BatchNorm2d(512),
                nn.ReLU(True), # out: 512 x 4 x 4 # (input size-1)*s -2p + k
                
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True), # out: 256 x 8 x 8
                
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True), # out: 128 x 16 x 16
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True), # out: 64 x 32 x 32
                
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()) # out: 3 x 64 x 64

# Device setting
D = D.to(device)
G = G.to(device)

def main():

    # Binary cross entropy loss and optimizer
    criterion = F.binary_cross_entropy
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    # Start training
    epochs = 25
    total_step = len(train_dl)
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_dl):
            images = images.reshape(batch_size, -1).to(device)
            
            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
            
            # Backprop and optimize
            reset_grad()
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                            real_score.mean().item(), fake_score.mean().item()))
        
        # Save real images
        if (epoch+1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join('/samples', 'real_images.png'))
        
        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join('/samples', 'fake_images-{}.png'.format(epoch+1)))

if __name__ == "__main__":
    main()