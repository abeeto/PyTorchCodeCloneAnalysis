""" main.py
"""
import argparse
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image

from loader import get_loader, denorm
from models.model import NetD, NetG

parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--dataset', type=str, default='../data/celeba')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--z_num', type=int, default=64)
parser.add_argument('--n_num', type=int, default=64)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--final_epoch', type=int, default=100)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--image_path', type=str, default='images_n')
parser.add_argument('--model_path', type=str, default='chkpts_n')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--up_mode', type=str, default='nearest')
parser.add_argument('--norm_mode', type=str, default='instance')
config = parser.parse_args()

# manual seed
torch.cuda.manual_seed(1502)
# data_loader
loader = get_loader(config)
# network
net_g = NetG(config).cuda()
net_d = NetD(config).cuda()
print(net_d)
print(net_g)
# criterion
bce = torch.nn.BCELoss().cuda()
# optimizer
opt_g = Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
# fixed
fixed = Variable(torch.Tensor(64, 64)).cuda()
fixed.data.normal_(0.0, 1.0)

def train(epoch):
    """ train """

    for idx, (image, _) in enumerate(loader):

        # update net d
        net_d.zero_grad()
        net_g.zero_grad()

        real = Variable(image).cuda()
        noise = Variable(torch.Tensor(image.size(0), config.z_num)).cuda()
        noise.data.normal_(0.0, 1.0)

        fake = net_g(noise)
        fake_d = net_d(fake.detach())
        real_d = net_d(real)
        real_label = Variable(torch.ones(real_d.size())).cuda()
        fake_label = Variable(torch.zeros(fake_d.size())).cuda()
        cost_d = bce(real_d, real_label) + bce(fake_d, fake_label)
        cost_d.backward()
        opt_d.step()

        # update net g
        fake_g = net_d(fake)
        real_label = Variable(torch.ones(fake_g.size())).cuda()
        cost_g = bce(fake_g, real_label)
        cost_g.backward()
        opt_g.step()

    # log
    fake_fixed = net_g(fixed)
    torch.save(net_g.state_dict(), '{0}/G_{1}.pth'.format(config.model_path, epoch))
    torch.save(net_d.state_dict(), '{0}/D_{1}.pth'.format(config.model_path, epoch))
    save_image(denorm(fake_fixed).data, '{0}/fixed_{1}.png'.format(config.image_path, epoch))
    save_image(denorm(fake).data, '{0}/fake_{1}.png'.format(config.image_path, epoch))


if __name__ == '__main__':

    os.makedirs(config.image_path, exist_ok=True)
    os.makedirs(config.model_path, exist_ok=True)

    for epoch in range(config.start_epoch, config.final_epoch):
        train(epoch)
