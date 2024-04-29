#!/usr/bin/env python
##########################################################
# File Name: load_data.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-05-10 14:09:39
##########################################################

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from illu_data.illu_dataset import IlluDataset

class Opt:
    def __init__(self):
        self.__dict__ = {"cuda": True, 
                         "imageSize": 224,
                         "orgSize": 672,
                         "dataset" : "box",
                         "batchSize" : 20,
                         "workers" : 20,
                         "dataroot" : "/data3/lzh/10000x672x672_box_diff/",
                         #"dataroot" : "/data3/lzh/torch_data",
                         "niter" : 100000,
                         "lr" : 1e-3,
                         "beta1" : 0.9,
                         "outf" : "snapshots",
                }

opt = Opt()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['box', 'Diamond', 'torus']:
    # folder dataset
    dataset = IlluDataset(root=opt.dataroot,
                        transform=transforms.Compose([
                    #    transforms.RandomCrop((opt.imageSize, opt.imageSize)),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.0,) * 30, (256.0,) * 30),
                        ]),
                        target_transform = transforms.Compose([
                            transforms.ToTensor(),
                         #   transforms.Normalize((0.0,) * 3, (256.0,) * 3),
                        ]))
#elif opt.dataset == 'fake':
#    dataset = dset.FakeData(image_size=(3, opt.orgSize, opt.orgSize),
#                            transform=transforms.Compose([
#                                transforms.RandomCrop((opt.imageSize, opt.imageSize)),
#                                transforms.ToTensor(),
#                            ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
#device = torch.cuda.device(0)
#ngpu = int(opt.ngpu)
#nz = int(opt.nz)
#ngf = int(opt.ngf)
#ndf = int(opt.ndf)
#nc = 3
ngpu = 2


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

from model.simple_resnet import Generator 
from model.parrel_resnet import Discriminator

netG = Generator(2, [3,3,3], 30).to(device)
#if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))
print(netG)

#nn.Module().to
netD = Discriminator(ngpu).to(device)
#netD.apply(weights_init)
#if opt.netD != '':
#    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
mse = nn.MSELoss(size_average = False)

#fixed_noise = torch.rand(1, 3, opt.imageSize, opt.imageSize, device = device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr*2, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data_label in enumerate(dataloader):
        data, groundtruth = data_label
        #print(data.shape)
        #print(label.shape)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = groundtruth.to(device)
        noise = data.to(device)
        fake = netG(noise)

        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(torch.cat([real_cpu, fake.detach()], dim = 1))
      
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        #noise = torch.randn(batch_size, nz, 1, 1, device=device)
        #noise = torch.rand(batch_size, 3, opt.imageSize, opt.imageSize, device = device)
        label.fill_(fake_label)
        output = netD(torch.cat([fake.detach(), real_cpu], dim = 1))
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(torch.cat([fake, real_cpu], dim = 1))
        D_G_z2 = output.mean().item()

        errG = criterion(output, label)
        #errG.backward()
        mseG = mse(fake, real_cpu) / batch_size 
        
        #if mseG > 50000:
        #    mseG.backward()
        #else:
        #    errG.backward()
        lossG = errG * 1000 + mseG 
        #lossG.backward()
        #mseG.backward()

        optimizerG.step()

        if i % 5 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f loss_MSE: %.4f D(x): %.4f D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), mseG.item(), D_x, D_G_z1))
           
    #if i % 5000 == 0:
    vutils.save_image(real_cpu,
            '%s/real_samples.png' % opt.outf,
            normalize=True)
    #fake = netG(fixed_noise)
    vutils.save_image(fake.detach().clamp(0.0, 1.0),
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=True)

    # do checkpointing
    if epoch % 20 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
