import random
import os.path
import numpy as np 

import torch
import torchvision
import torch.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import G, D
from dataset import get_loader


class Pix2Pix:
    def __init__(self, batch_size, epoch_iter, lr, src_path, trg_path, sample_img_path, save_model_path, restore_D_model_path, restore_G_model_path, gpu):
        self.batch_size = batch_size
        self.epoch_iter = epoch_iter
        self.lr = lr

        self.src_path = src_path
        self.trg_path = trg_path
        self.sample_img_path = sample_img_path
        self.save_model_path = save_model_path
        self.restore_D_model_path = restore_D_model_path
        self.restore_G_model_path = restore_G_model_path

        self.D = D()
        self.G = G()
        self.d_step = 1
        self.g_step = 1

        self.gpu = gpu

        self.transform = transforms.Compose([transforms.ToTensor()])
                                            #  transforms.Normalize((0.485, 0.456, 0.406),
                                            #                       (0.229, 0.224, 0.225))])

    def load_dataset(self):
        src_data = dset.ImageFolder(self.src_path, self.transformations)
        self.src_loader = DataLoader(src_data, batch_size=self.batch_size)
        trg_data = dset.ImageFolder(self.trg_path, self.transformations)
        self.trg_loader = DataLoader(trg_data, batch_size=self.batch_size)

    def train(self):
        data_loader = get_loader(self.batch_size, self.src_path, self.trg_path, self.transform)
        print('Dataset Load Success!')

        if len(self.restore_G_model_path):
            self.D.load_state_dict(torch.load(self.restore_D_model_path))
            self.G.load_state_dict(torch.load(self.restore_G_model_path))
            print('Pretrained model load success!')

        D_adam = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        G_adam = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))

        if self.gpu:
            self.D = self.D.cuda()
            self.G = self.G.cuda()
            ones, zeros = Variable(torch.ones(self.batch_size, 1, 30, 30).cuda()), Variable(torch.zeros(self.batch_size, 1, 30, 30).cuda())
            BCE_loss = nn.BCELoss().cuda()
            L1_loss = nn.L1Loss().cuda()
            MSE_loss = nn.MSELoss().cuda()
        else:
            ones, zeros = Variable(torch.ones(self.batch_size, 1, 30, 30)), Variable(torch.zeros(self.batch_size, 1, 30, 30))
            BCE_loss = nn.BCELoss()
            L1_loss = nn.L1Loss()
            MSE_loss = nn.MSELoss()

        self.D.train()
        self.G.train()
        print('Training Start')
        for epoch in range(3, self.epoch_iter):
            for step, (src, trg) in enumerate(data_loader):
                for d_i in range(self.d_step):
                    src, trg = iter(data_loader).next()
                    src_data = src.cuda()
                    trg_data = trg.cuda()

                    self.D.zero_grad()
                    self.G.zero_grad()

                    src_input = Variable(src_data.cuda())
                    trg_input = Variable(trg_data.cuda())

                    src_generated = self.G(src_input)

                    D_src_generated = self.D(src_generated, trg_input)
                    D_trg_input = self.D(trg_input, trg_input)

                    # training D
                    D_fake_loss = BCE_loss(D_src_generated, zeros)
                    D_real_loss = BCE_loss(D_trg_input, ones)
                    D_loss = D_fake_loss + D_real_loss
                    D_loss.backward(retain_graph=True)
                    D_adam.step()
                
                for p in self.D.parameters():
                    p.requires_grad = False
                
                for g_i in range(self.g_step):
                    # training G
                    G_fake_loss = BCE_loss(D_src_generated, ones)
                    G_distance_loss = MSE_loss(src_generated, trg_input) * 100
                    G_loss = G_fake_loss + G_distance_loss
                    G_loss.backward(retain_graph=True)
                    G_adam.step()
                    
                    
                for p in self.D.parameters():
                    p.requires_grad = True

                # logging losses
                if step % 20 == 0:
                    print(f"Epoch: {epoch} & Step: {step} => D-fake Loss: {D_fake_loss.data}, D-real Loss: {D_real_loss.data}, G Loss: {G_loss.data}")
                    
                # save sample images 
                if step % 50 == 0:
                    vutils.save_image(src_data[0], os.path.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-src_input.jpg'))
                    vutils.save_image(trg_data[0], os.path.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-trg_input.jpg'))
                    vutils.save_image(src_generated.data[0], os.path.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-generated.jpg'))

            # save model
            torch.save(self.D.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'D' + '.pth'))
            torch.save(self.G.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'G' + '.pth'))
                
