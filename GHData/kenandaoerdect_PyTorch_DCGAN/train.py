import argparse
import os
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from model import NetD, NetG


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='imgs/', help='folder to output images')
parser.add_argument('--model_path', default='saved_model/', help='folder to model checkpoints')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
if not os.path.exists(opt.model_path):
    os.makedirs(opt.model_path)

for epoch in range(1, opt.epoch + 1):
    for i, (imgs,_) in enumerate(dataloader):
        optimizerD.zero_grad()
        imgs=imgs.to(device)
        output = netD(imgs)
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()

        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise=noise.to(device)
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        optimizerG.zero_grad()
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D:%.3f Loss_G:%.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    if epoch % 10 == 0:
        torch.save(netG, '%s/netG-epoch_%03d.pth' % (opt.model_path, epoch))
        torch.save(netD, '%s/netD-epoch_%03d.pth' % (opt.model_path, epoch))

torch.save(netG, '%s/netG-epoch_%03d.pth' % (opt.model_path, opt.epoch))
torch.save(netD, '%s/netD-epoch_%03d.pth' % (opt.model_path, opt.epoch))