from __future__ import print_function
import argparse
import os
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./result', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed (-1: random seed)')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed == -1:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'stl10':
    dataset = dset.STL10(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(opt.imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].view(-1, num_classes, 1, 1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.weight.data.h
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.cv1 = nn.ConvTranspose2d(nz+10, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        # state size. (ngf*8) x 4 x 4
        self.cv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        # state size. (ngf*4) x 8 x 8
        self.cv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        # state size. (ngf*2) x 16 x 16
        self.cv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x 32 x 32
        self.cv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

    def forward(self, input, label):
        input_label = torch.cat((input, label), dim=1)

        hidden = F.relu(self.bn1(self.cv1(input_label)))
        hidden = F.relu(self.bn2(self.cv2(hidden)))
        hidden = F.relu(self.bn3(self.cv3(hidden)))
        hidden = F.relu(self.bn4(self.cv4(hidden)))
        output = F.tanh(self.cv5(hidden))
        return output

    def initialization(self):
        nn.init.kaiming_normal_(self.cv1.weight)
        nn.init.normal_(self.bn1.weight, 1.0, 0.02)
        nn.init.constant_(self.bn1.bias, 0.)

        nn.init.kaiming_normal_(self.cv2.weight)
        nn.init.normal_(self.bn2.weight, 1.0, 0.02)
        nn.init.constant_(self.bn2.bias, 0.)

        nn.init.kaiming_normal_(self.cv3.weight)
        nn.init.normal_(self.bn3.weight, 1.0, 0.02)
        nn.init.constant_(self.bn3.bias, 0.)

        nn.init.kaiming_normal_(self.cv4.weight)
        nn.init.normal_(self.bn4.weight, 1.0, 0.02)
        nn.init.constant_(self.bn4.bias, 0.)

        nn.init.kaiming_normal_(self.cv5.weight)


netG = Generator(ngpu).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.cv0 = nn.Conv2d(nc + 10, ndf, 4, 2, 1, bias=False)
        # state size. (ndf) x 32 x 32
        self.cv1 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf*2) x 16 x 16
        self.cv2 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf*4) x 8 x 8
        self.cv3 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf*8) x 4 x 4
        self.cv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input, label):
        # label = label.view(-1, 10, 1, 1)
        label = label.expand((50, 10, 64, 64))
        input_label = torch.cat((input, label), dim=1)

        hidden = F.leaky_relu(self.cv0(input_label), negative_slope=0.2)
        hidden = F.leaky_relu(self.bn1(self.cv1(hidden)), negative_slope=0.2)
        hidden = F.leaky_relu(self.bn2(self.cv2(hidden)), negative_slope=0.2)
        hidden = F.leaky_relu(self.bn3(self.cv3(hidden)), negative_slope=0.2)
        output = self.cv4(hidden)    # output = F.sigmoid(self.cv4(hidden))
        return output.view(-1, 1).squeeze(1)

    def initialization(self):
        nn.init.kaiming_normal_(self.cv0.weight, a=0.2)

        nn.init.kaiming_normal_(self.cv1.weight, a=0.2)
        nn.init.normal_(self.bn1.weight, 1., 0.02)
        nn.init.constant_(self.bn1.bias, 0.)

        nn.init.kaiming_normal_(self.cv2.weight, a=0.2)
        nn.init.normal_(self.bn2.weight, 1., 0.02)
        nn.init.constant_(self.bn2.bias, 0.)

        nn.init.kaiming_normal_(self.cv3.weight, a=0.2)
        nn.init.normal_(self.bn3.weight, 1., 0.02)
        nn.init.constant_(self.bn3.bias, 0.)

        nn.init.kaiming_normal_(self.cv4.weight, a=0.2)


netD = Discriminator(ngpu).to(device)
# netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.MSELoss()    # criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
# noise_label = torch.randint(10, (opt.batchSize,), dtype=torch.long, device=device)
noise_label = [i for i in range(10)] * 5
noise_label = torch.tensor(noise_label, dtype=torch.long, device=device)

oh_noise_label = one_hot_embedding(noise_label).to(device)

real_target = 1
fake_target = 0

# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

for epoch in range(opt.niter):
    for itr, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real = data[0].to(device)
        real_label = data[1].to(device)
        oh_real_label = one_hot_embedding(real_label).to(device)

        batch_size = real.size(0)
        target = torch.full((batch_size,), real_target, device=device)

        output = netD(real, oh_real_label)
        errD_real = criterion(output, target)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_label = torch.randint(10, (batch_size,), dtype=torch.long, device=device)
        oh_fake_label = one_hot_embedding(fake_label).to(device)

        fake = netG(noise, oh_fake_label)
        target.fill_(fake_target)
        output = netD(fake.detach(), oh_fake_label)

        errD_fake = criterion(output, target)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        target.fill_(real_target)  # fake labels are real for generator cost
        output = netD(fake, oh_fake_label)
        errG = criterion(output, target)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f D(x): %.3f D(G(z)): %.3f / %.3f'
              % (epoch + 1, opt.niter, itr + 1, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if epoch == 0:
        vutils.save_image(real,
                '%s/real_samples.png' % opt.outf,
                          normalize=True, nrow=10)
    fake = netG(fixed_noise, oh_noise_label)
    vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch + 1),
            normalize=True, nrow=10)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))
