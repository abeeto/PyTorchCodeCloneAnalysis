import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import DatasetFromChictopia
from models import G, D, weights_init

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', action='store_true', help='use cuda gpu')
parser.add_argument('-b', '--batch_size', help='select batch size(default is 16)', default=16, type=int)
parser.add_argument('-n', '--num_workers', help='select batch size(default is 4)', default=4, type=int)
parser.add_argument('-d', '--dir', help='image directory(default is ./Chictopia)', default='./Chictopia')
opt = parser.parse_args()

print('===> Loading datasets')
root_path = opt.dir
train_set = DatasetFromChictopia(image_dir=os.path.join(root_path, 'train'), mode='train')
test_set = DatasetFromChictopia(image_dir=os.path.join(root_path, 'test'), mode='test')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                  shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                 shuffle=False)

print('===> Building model')
netG = G(3, 3, 64)
netG.apply(weights_init)
netD = D(3, 3, 64)
netD.apply(weights_init)

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

real_A = torch.FloatTensor(16, 3, 256, 256)
real_B = torch.FloatTensor(16, 3, 256, 256)
label = torch.FloatTensor(16)
real_label = 1
fake_label = 0

real_A = Variable(real_A)
real_B = Variable(real_B)
label = Variable(label)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()
    criterion_l1 = criterion_l1.cuda()
    criterion_mse = criterion_mse.cuda()
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train with real
        netG.eval()
        netD.train()
        optimizerD.zero_grad()
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_B.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        output = netD(torch.cat((real_A, real_B), 1))
        label.data.resize_(output.size()).fill_(real_label)
        err_d_real = criterion(output, label)
        err_d_real.backward()
        d_x_y = output.data.mean()

        # train with fake
        fake_b = netG(real_A)
        output = netD(torch.cat((real_A, fake_b.detach()), 1))
        label.data.resize_(output.size()).fill_(fake_label)
        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_x_gx = output.data.mean()

        err_d = (err_d_real + err_d_fake) / 2.0
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ###########################
        netG.train()
        netD.eval()
        optimizerG.zero_grad()
        output = netD(torch.cat((real_A, fake_b), 1))
        label.data.resize_(output.size()).fill_(real_label)
        err_g = criterion(output, label) + 100 * criterion_l1(fake_b, real_B)
        err_g.backward()
        d_x_gx_2 = output.data.mean()
        optimizerG.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}".format(
            epoch, iteration, len(training_data_loader), err_d.data[0], err_g.data[0], d_x_y, d_x_gx, d_x_gx_2))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        prediction = netG(input)
        mse = criterion_mse(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
    net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
    torch.save(netG.state_dict(), net_g_model_out_path)
    torch.save(netD.state_dict(), net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint"))


for epoch in range(1, 100 + 1):
    train(epoch)
    test()
    if epoch % 50 == 0:
        checkpoint(epoch)
