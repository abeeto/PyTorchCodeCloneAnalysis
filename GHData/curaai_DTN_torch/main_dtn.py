import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import *
from data import *
from dataset import get_loader


model_path = 'model/vgg16bn.pth'

# parameter for adam optimizer
g_lr = 0.0001
d_lr = 0.0001
beta1 = 0.9
beta2 = 0.99

gpu_mode = True

# network init 
g = G()
d = D()
e = E(model_path)

batch_size = 50
n_epoch = 10000

d_steps = 5
g_steps = 1

alpha = 15
beta = 15

g_adam = optim.Adam(g.parameters(), lr=g_lr, betas=(beta1, beta2))
d_adam = optim.Adam(d.parameters(), lr=d_lr, betas=(beta1, beta2))

if gpu_mode:
    g = g.cuda()
    d = d.cuda()
    e = e.cuda()

    BCE_loss = nn.BCELoss().cuda()
else:
    BCE_loss = nn.BCELoss()      
MSE_loss = nn.MSELoss()

animal_dir = "data/animal"
pokemon_dir = "data/pokemon"

transformations = transforms.Compose([
    transforms.Scale(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

data_loader = get_loader(batch_size, animal_dir, pokemon_dir, transformations)

if gpu_mode:
    ones, zeros = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())
else:
    y_real_, y_fake_ = Variable(torch.ones(batch_size, 1)), Variable(torch.zeros(batch_size, 1))

g.train()
d.train()
e.eval()

for p in e.parameters():
    p.required_grad = False


for epoch in range(n_epoch):
    for step, (animal, pokemon) in enumerate(data_loader):

        d_adam.zero_grad()
        g_adam.zero_grad()

        src_input = Variable(animal)
        trg_input = Variable(pokemon)

        # f(x)
        src_encoded = e.model(src_input)
        trg_encoded = e.model(trg_input)

        # G(f(x))
        src_generated = g.model(src_encoded)
        trg_generated = g.model(trg_encoded)

        #f(G(f(x)))
        src_encode_generated = e.model(src_encoded)

        # D losses d_i is in paper 
        d1_src_loss = BCE_loss(src_generated, zeros)
        d2_trg_loss = BCE_loss(trg_generated, zeros)
        d3_trg_loss = BCE_loss(trg_input, ones)
        loss_D = d1_src_loss + d2_trg_loss + d3_trg_loss
        d_adam.step()
        for p in d.parameters():
            p.requires_grad = False

        # L_constancy 
        L_const = MSE_loss(src_encoded, src_encode_generated) * alpha
        L_const.backward(retain_variables=True)

        # L_TID 
        L_tid = MSE_loss(trg_input, trg_generated) * beta
        L_tid.backward(retain_variables=True)

        # G loss
        g_src_loss = BCE_loss(src_generated, ones)
        g_trg_loss = BCE_loss(trg_generated, ones)
        loss_G = g_src_loss + g_trg_loss
        g_adam.step()
        for p in d.parameters():
            p.requires_grad = True

        if step % 10 == 0:
            vutils.save_image(src_generated.data[0], os.path.join('result', f'{epoch}_{step}.jpg'))
        
    if epoch % 3 == 0:
        torch.save(g.state_dict(), os.path.join('model/' + str(epoch)+ 'G.pth'))
        torch.save(d.state_dict(), os.path.join('model/' + str(epoch)+ 'D.pth'))
        torch.save(e.state_dict(), os.path.join('model/' + str(epoch)+ 'E.pth'))