import torch.nn as nn
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")

batch_size = 32
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         strd=[0.5,0.5,0.5])
])

mnist_data = torchvision.datasets.MNIST("./mnist_data",train=True,download=True,transform=transform)
dataloader = torch.utils.data.DataLoader(
    datasets=mnist_data,
    batch_size=batch_size,
    shuffle=True
)

image_size = 784
hidden_size = 256

D = nn.Sequential(
    nn.Linear(image_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)

latent_size = 64
G = nn.Sequential(
    nn.Linear(latent_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,image_size),
    nn.Tanh()
)

D = D.to(device)
G = G.to(device)
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0002)

#下面开始训练，明天后再搞
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
total_step = len(dataloader)
num_epochs = 200
for epoch in range(num_epochs):
    for i,(images,_) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.reshape(batch_size,image_size).to(device)

        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        outputs = D(images)
        d_loss_real = loss_fn(outputs,real_labels)
        real_score = outputs

        #生成fake iamges
        z = torch.randn(batch_size,latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = loss_fn(outputs,fake_labels)
        fake_score = outputs

        #开始优化discriminator
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        #开始优化generator
        z = torch.randn(batch_size,latent_size).to(device)
        fake_iamges = G(z)
        outputs = D(fake_iamges)
        g_loss = loss_fn(outputs,real_labels)

        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if i%1000 == 0:
            print("Epoch [{}/{}], Step [{}/{}],d_loss:{},g_loss:{},D(x):{},D(G(z)):{}"
                  .format(epoch,num_epochs,i,total_step,d_loss.item(),g_loss.item(),
                          real_score.mean().item(),fake_score.mean().item()))

#训练完成以后

import matplotlib.pyplot as plt
z = torch.randn(1,latent_size).to(device)
fake_images = G(z).view(28,28).data.cpu().numpy()
plt.imshow(fake_images)

#真实图片：
plt.imshow(images[0].view(28,28).data.cpu().numpy())
