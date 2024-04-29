import os
import torch
import torch.nn as nn
from net import Discriminator, Generator
from dataloader import dataloader
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import to_img

if not os.path.exists("./data"):
    os.mkdir("./data")

if not os.path.exists("./result"):
    os.mkdir("./result")

if not os.path.exists("./model"):
    os.mkdir("./model")


batch_size = 64
z_dimension = 100
num_epoch = 1000

D = Discriminator()
G = Generator(z_dimension)

# if torch.cuda.is_available:
D = D.cuda()
G = G.cuda()

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if(i+1) % 100 == 0:
            print("Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}, D_real: {:.6f}, D_fake: {:.6f}".format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(), real_scores.data.mean(), fake_scores.data.mean()
            ))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, "./result/real_images.png")
    
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, "./result/fake_images-{}.png".format(epoch+1))

torch.save(G.state_dict(), './model/generator.pth')
torch.save(D.state_dict(), "./model/discriminator.pth")



