# GAN 生成式对抗网络 Generative Adversarial Network
# 用MNIST作为数据集，尝试生成假的MNIST图片
# 模型要有Discriminator和Generator
# 判别器要尽可能判断出假的图片，生成器则要尽可能骗过判别器
# 判别器和生成器都用简单的Linear搭建
# 在多轮epoch之后，对生成器生成精准的图片进行“鼓励”

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as tud
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

batch_size = 32
image_size = 28 * 28
hidden_size = 256
# 用于生成fake image
latent_size = 64
num_epochs = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_data = datasets.MNIST(".data", train=True, download=True, transform=transform)
dataloader = tud.DataLoader(mnist_data, batch_size, shuffle=True)

# Discriminator 生成器
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
).to(device)

# Generator 判别器
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
).to(device)

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def clear_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# 每五次用真图片对判别器进行训练
# 每次用假图片对生成器进行训练
for e in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.reshape(batch_size, image_size).to(device)
        # 生成图片对应的标签，真图片为1，假图片为0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        
        # 先用判别器对真图片进行训练
        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs.mean().item()

        # 开始生成假的图片
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach()) # 防止生成器的导数影响
        if e > 5:
            fake_labels[outputs>0.9999] = 1 # 多轮之后，判别器已经成熟，所以此时能骗过判别器的都应该被鼓励
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs.mean().item()

        # 开始优化判别器
        d_loss = d_loss_fake + d_loss_real
        clear_grad()
        d_loss.backward()
        d_optimizer.step()

        # 开始优化生成器
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels)
        clear_grad()
        g_loss.backward()
        g_optimizer.step()

    print("Epoch: {}, d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}"
                .format(e, d_loss.item(), g_loss.item(), real_score, fake_score))

for i in range(5):
    plt.figure()
    z = torch.randn(1, latent_size).to(device)
    fake_images = G(z).view(28, 28).data.cpu().numpy()
    fake_images = (fake_images+1)/2
    plt.imshow(fake_images, cmap=plt.cm.gray) # 黑白显示
    plt.show()
