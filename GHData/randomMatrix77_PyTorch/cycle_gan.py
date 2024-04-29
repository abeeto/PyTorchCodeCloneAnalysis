import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Discriminator

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn1b = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2a = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2b = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3a = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3c = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        self.conv4a = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size = 3, stride =1, padding = 1)
        self.bn4b = nn.BatchNorm2d(512)
        self.conv4c = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn4c = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2, return_indices = True)

        self.conv1 = nn.Sequential(self.conv1a, self.bn1a,
                                  self.conv1b, self.bn1b, self.act1)
        self.conv2 = nn.Sequential(self.conv2a, self.bn2a,
                                  self.conv2b, self.bn2b, self.act2)
        self.conv3 = nn.Sequential(self.conv3a, self.bn3a,
                                   self.conv3b, self.bn3b,
                                   self.conv3c, self.bn3c, self.act3)
        self.conv4 = nn.Sequential(self.conv4a, self.bn4a,
                                   self.conv4b, self.bn4b,
                                   self.conv4c, self.bn4c, self.act4)

    def forward(self, x):

        idx = []
        
        out = self.conv1(x)
        out, idx1 = self.pool(out)
        idx.append(idx1)
        
        out = self.conv2(out)
        out, idx2 = self.pool(out)
        idx.append(idx2)

        out = self.conv3(out)
        out, idx3 = self.pool(out)
        idx.append(idx3)

        out = self.conv4(out)
        out, idx4 = self.pool(out)
        idx.append(idx4)
        
        return out, idx

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        #Unpool
        self.conv4c = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn4c = nn.BatchNorm2d(512)
        self.act4c = nn.ReLU()
        self.conv4b = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn4b = nn.BatchNorm2d(512)
        self.act4b = nn.ReLU()
        self.conv4a = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn4a = nn.BatchNorm2d(256)
        self.act4a = nn.ReLU()
        self.stage4 = nn.Sequential(self.conv4c, self.bn4c, self.act4c,
                                    self.conv4b, self.bn4b, self.act4b,
                                    self.conv4a, self.bn4a, self.act4a)

        #unpool
        self.conv3c = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3c = nn.BatchNorm2d(256)
        self.act3c = nn.ReLU()
        self.conv3b = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3b = nn.BatchNorm2d(256)
        self.act3b = nn.ReLU()
        self.conv3a = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3a = nn.BatchNorm2d(128)
        self.act3a = nn. ReLU()
        self.stage3 = nn.Sequential(self.conv3c, self.bn3c, self.act3c,
                                    self.conv3b, self.bn3b, self.act3b,
                                    self.conv3a, self.bn3a, self.act3a)

        #unpool
        self.conv2b = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2b = nn.BatchNorm2d(128)
        self.act2b = nn.ReLU()
        self.conv2a = nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn2a = nn.BatchNorm2d(64)
        self.act2a = nn. ReLU()
        self.stage2 = nn.Sequential(self.conv2b, self.bn2b, self.act2b,
                                    self.conv2a, self.bn2a, self.act2a)

        #unpool
        self.conv1b = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn1b = nn.BatchNorm2d(64)
        self.act1b = nn.ReLU()
        self.conv1a = nn.Conv2d(64, 1, kernel_size = 3, stride = 1, padding = 1)
        self.bn1a = nn.BatchNorm2d(1)
        self.act1a = nn. ReLU()
        self.stage1 = nn.Sequential(self.conv1b, self.bn1b, self.act1b,
                                    self.conv1a, self.bn1a, self.act1a)

        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x, idx):

        out = self.unpool(x, idx[3])
        out = self.stage4(out)

        out = self.unpool(out, idx[2])
        out = self.stage3(out)

        out = self.unpool(out, idx[1])
        out = self.stage2(out)

        out = self.unpool(out, idx[0])
        out = self.stage1(out)

        return out

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.E = Encoder()
        self.D = Decoder()

# Generator

class Generator(nn.Module):

    def __init__(self):

        super(UNet, self).__init__()

        self.conv1a = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2b = nn.BatchNorm2d(128)

        self.middle = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

        self.conv1 = nn.Sequential(self.conv1a, self.bn1a, self.act,
                                   self.conv1b, self.bn1b, self.act)

        self.conv2 = nn.Sequential(self.conv2a, self.bn2a, self.act,
                                   self.conv2b, self.bn2b, self.act)

        self.conv3a = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn3b = nn.BatchNorm2d(32)

        self.conv4a = nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn4a = nn.BatchNorm2d(32)
        self.conv4b = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn4b = nn.BatchNorm2d(32)

        self.conv5a = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn5a = nn.BatchNorm2d(16)
        self.conv5b = nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1)
        self.bn5b = nn.BatchNorm2d(1)

        self.up1 = nn.Sequential(self.conv3a, self.bn3a, self.act,
                                 self.conv3b, self.bn3b, self.act)

        self.up2 = nn.Sequential(self.conv4a, self.bn4a, self.act,
                                 self.conv4b, self.bn4b, self.act)

        self.up3 = nn.Sequential(self.conv5a, self.bn5a, self.act,
                                 self.conv5b, self.bn5b, self.act)

    def forward(self, x):

        out1 = self.conv1(x)
        out1 = self.pool(out1)

        out2 = self.conv2(out1)
        out2 = self.pool(out2)

        out3 = self.middle(out2)
        out3 = self.pool(out3)

        out4 = self.upsample(out3)
        out4 = torch.cat((out2, out4), dim = 1)
        out4 = self.up1(out4)

        out5 = self.upsample(out4)
        out5 = torch.cat((out1, out5), dim = 1)
        out5 = self.up2(out5)

        out6 = self.upsample(out5)
        out6 = self.up3(out6)

        return out6

gen_a = Generator()
gen_b = Generator()

disc_a = Discriminator()
disc_b = Discriminator()

gen_optim = torch.optim.Adam(itertools.chain(gen_a.parameters(), gen_b.parameters()), lr)
disc_a_optim = torch.optim.Adam(disc_a.parameters(), lr)
disc_b_optim = torch.optim.Adam(disc_b.parameters(), lr)

mse = nn.MSELoss()
l1 = nn.L1Loss()

lam = 10

for i in range(epochs):

    gen_optim.zero_grad()

    fakeB = gen_a(x)
    discB = disc_b(fakeB)
    g_a_loss = mse(discB, torch.ones_like(discB))

    recA = gen_b(fakeB)
    l1_a = l1(recA, x) * lam

    gen_a_loss = g_a_loss + l1_a

    fakeA = gen_b(y)
    discA = disc_a(fakeA)
    g_b_loss = mse(discA, torch.ones_like(discA))

    recB = gen_a(fakeA)
    l1_b = l1(recB, y) * lam

    gen_b_loss = g_b_loss + l1_b

    g_loss = gen_a_loss + gen_b_loss

    g_loss.backward()
    gen_optim.step()

    # Discriminator A

    disc_b_optim.zero_grad()

        #real B
    discB_r = disc_b(x)
    d_b_loss_r = mse(discB_r, torch.ones_like(discB_t))

        #fake B
    discB_f = disc_b(fakeB)
    d_b_loss_f = mse(discB_f, torch.zeros_like(discB_f))

    discB_loss = d_b_loss_r + d_b_loss_f

    discB_loss.backward()
    disc_b_optim.step()

    # Discriminator B

    disc_a_optim.zero_grad()
        #real A
    discA_r = disc_a(y)
    d_a_loss_r = mse(discA_r, torch.ones_like(discA_r))
        #fake A
    discA_f = disc_a(fakeA)
    d_a_loss_f = mse(discA_f, torch.ones_like(discA_f))

    discA_loss = d_a_loss_r + d_a_loss_f

    discA_loss.backward()
    disc_b_optim.step()

    

    







    
