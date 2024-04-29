# DCGAN Deep Convolutional GAN
# 用更加复杂的卷积神经网络构建模型
# 同样是判别器和生成器相互判别
# 最终模拟MNIST生成新的图片


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as tud
from torchvision import transforms, models, datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 24
torch.random.manual_seed(SEED)

image_size= 28*28
batch_size= 32
latent_size = 100 # latent vector的大小
ngf = 16 # generator feature map size
ndf = 16 # discriminator feature map size
num_channels = 1 # color channels
learning_rate = 0.0002
beta1 = 0.5
num_epochs = 20


dataset = torchvision.datasets.MNIST(root=".data", transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

'''
# 成组展示图片
real_batch=next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis=("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()
'''

# 初始化weight，不同函数有不同的初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is batch_size * 1 * 28 * 28
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
            #       dilation=1, groups=1, bias=True, padding_mode='zeros')
            # H_out = [H_in + 2×padding[0] − (kernel_size[0]−1)×dilation[0] − 1] / stride[0] + 1
            # W_out = [W_in + 2×padding[1] − (kernel_size[1]−1)×dilation[1] − 1] / stride[1] + 1

            # batch_size * 1 * 28 * 28 -> batch_size * 16 * 14 * 14 
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # batch_size * 16 * 14 * 14 -> batch_size * 32 * 7 * 7 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # batch_size * 32 * 7 * 7 -> batch_size * 64 * 4 * 4
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # batch_size * 64 * 4 * 4 -> batch_size * 128 * 2 * 2 
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # batch_size * 128 * 2 * 2 -> batch_size * 1 * 4 * 4 
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is batch_size * 100 * 1 * 1
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
            #       output_padding=0, groups=1, bias=True, dilation=1)
            # H_out = (H_in−1)×stride[0] − 2×padding[0] + (kernel_size[0]−1)×dilation[0] + output_padding[0] + 1
            # W_out = (W_in−1)×stride[1] − 2×padding[1] + (kernel_size[1]−1)×dilation[1] + output_padding[1] + 1

            # batch_size * 100 * 1 * 1 -> batch_size * 128 * 4 * 4 
            nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # batch_size * 128 * 4 * 4 -> batch_size * 64 * 7 * 7 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # batch_size * 64 * 7 * 7 -> batch_size * 32 * 14 * 14 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # batch_size * 32 * 14 * 14 -> batch_size * 16 * 28 * 28 
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # batch_size * 16 * 28 * 28 -> batch_size * num_channels * 28 * 28
            nn.ConvTranspose2d(ngf, num_channels, 3, 1, 1, bias=False),
            nn.Tanh()
            # batch_size * num_channels * 28 * 28
        )

    def forward(self, input):
        return self.main(input)


# 建立模型并且初始化
D = Discriminator().to(device)
D.apply(weights_init)
G = Generator().to(device)
G.apply(weights_init)

'''
# Print the model
print(D)
print(G)
'''

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))

G_losses = []
D_losses = []
for e in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        # 训练discriminator, maximize log(D(x)) + log(1-D(G(z)))
        
        # 首先训练真实图片
        D.zero_grad()
        real_images = data.to(device)
        b_size = real_images.size(0)
        label = torch.ones(b_size).to(device)
        output = D(real_images).view(-1)
        
        real_loss = loss_fn(output, label)
        real_loss.backward()
        D_x = output.mean().item()
        
        
        # 然后训练生成的假图片
        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        fake_images = G(noise)
        label.fill_(0)
        output = D(fake_images.detach()).view(-1)
        fake_loss = loss_fn(output, label)
        fake_loss.backward()
        loss_D = real_loss + fake_loss
        d_optimizer.step()
        
        # 训练Generator
        G.zero_grad()
        label.fill_(1)
        output = D(fake_images).view(-1)
        loss_G = loss_fn(output, label)
        loss_G.backward()
        D_G_z = output.mean().item()
        g_optimizer.step()
        
        if i % 500 == 0:
            print("Epoch: {} Iter: {} Loss_D: {:.4f} Loss_G {:.4f} D(x): {:.4f} D(G(z)): {:.4f}"
                 .format(e, i/500, loss_D.item(), loss_G.item(), D_x, D_G_z))
        
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

real_batch, _ = next(iter(dataloader))
with torch.no_grad():
    fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)
    fake = G(fixed_noise).detach().cpu()


# Plot the real images
plt.figure(figsize=(28,28))
plt.subplot(1,2,1)
plt.axis=("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis=("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
plt.show()

