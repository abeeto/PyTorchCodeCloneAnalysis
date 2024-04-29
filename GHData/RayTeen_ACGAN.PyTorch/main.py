# pylint: disable=E1101
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F
from models import D, G

FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
FLAG.add_argument('--dataset', default='mnist', help='cifar10 | mnist.')
FLAG.add_argument('--dataroot', default='data', help='path to dataset.')
FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
FLAG.add_argument('--image_size', default=64, help='image size.')
FLAG.add_argument('--batch_size', default=64, help='batch size.')
FLAG.add_argument('--num_workers', default=10, help='num workers.')
FLAG.add_argument('--num_epoches', default=50, help='num workers.')
FLAG.add_argument('--nz', default=64, help='length of noize.')
FLAG.add_argument('--ndf', default=64, help='number of filters.')
FLAG.add_argument('--ngf', default=64, help='number of filters.')
opt = FLAG.parse_args()

os.makedirs('images', exist_ok=True)
os.makedirs('chkpts', exist_ok=True)

assert torch.cuda.is_available(), '[!] CUDA required!'

torch.cuda.manual_seed(opt.manual_seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True

tsfm=transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True, train=True, transform=tsfm)

loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

bce = nn.BCELoss().cuda()
cep = nn.CrossEntropyLoss().cuda()

opt.nc = 1 if opt.dataset == 'mnist' else 3

netd = D(ndf=opt.ndf, nc=opt.nc, num_classes=10).cuda()
netg = G(ngf=opt.ngf, nc=opt.nc, nz=opt.nz).cuda()

optd = optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))
optg = optim.Adam(netg.parameters(), lr=0.0002, betas=(0.5, 0.999))

embed = nn.Embedding(10, opt.nz).cuda()
label = Variable(torch.LongTensor([range(10)]*10)).view(-1).cuda()
fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
fixed.mul_(embed(label))

def denorm(x):
    return x * 0.5 + 0.5

def train(epoch):
    netg.train()
    netd.train()
    for _, (image, label) in enumerate(loader):
        #######################
        # real input and label
        #######################
        real_input = Variable(image).cuda()
        real_label = Variable(label).cuda()
        real_ = Variable(torch.ones(real_label.size())).cuda()

        #######################
        # fake input and label
        #######################
        noise = Variable(torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1)).cuda()
        fake_label = Variable(torch.LongTensor(opt.batch_size).random_(10)).cuda()
        noise.mul_(embed(fake_label))
        fake_ = Variable(torch.zeros(fake_label.size())).cuda()

        #######################
        # update net d
        #######################
        netd.zero_grad()
        fake_input = netg(noise)

        real_pred, real_cls = netd(real_input)
        fake_pred, fake_cls = netd(fake_input.detach())

        real_loss = bce(real_pred, real_) + cep(real_cls, real_label) * 10
        fake_loss = bce(fake_pred, fake_) + cep(fake_cls, fake_label) * 10
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optd.step()

        #######################
        # update net g
        #######################
        optg.zero_grad()
        fake_pred, fake_cls = netd(fake_input)
        real_ = Variable(torch.ones(fake_label.size())).cuda()
        g_loss = bce(fake_pred, real_) + cep(fake_cls, fake_label) * 10
        g_loss.backward()
        optg.step()

    #######################
    # save image pre epoch
    #######################
    utils.save_image(denorm(fake_input.data), f'images/fake_{epoch:03d}.jpg')
    utils.save_image(denorm(real_input.data), f'images/real_{epoch:03d}.jpg')

    #######################
    # save model pre epoch
    #######################
    torch.save(netg, f'chkpts/g_{epoch:03d}.pth')
    torch.save(netd, f'chkpts/d_{epoch:03d}.pth')


def test(epoch):
    netg.eval()

    fixed_input = netg(fixed)

    utils.save_image(denorm(fixed_input.data), f'images/fixed_{epoch:03d}.jpg', nrow=10)


if __name__ == '__main__':
    
    for epoch in range(opt.num_epoches):
        print(f'Epoch {epoch:03d}.')
        train(epoch)
        test(epoch)
