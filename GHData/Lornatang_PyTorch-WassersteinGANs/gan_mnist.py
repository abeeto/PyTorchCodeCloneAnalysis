# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.rmsprop import RMSprop

from wassersteinGANs.model.cnn import Discriminator
from wassersteinGANs.model.cnn import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str,
                    default="./datasets", help='path to datasets.')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=28,
                    help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 for adam. default=0.999')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=500, help="Train loop")
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument("--n_critic", type=int, default=5,
                    help='number of training steps for discriminator per iter')
parser.add_argument("--clip_value", type=float, default=0.01,
                    help='lower and upper clip value for disc. weights')
parser.add_argument('--outf', default='./imgs', help='folder to output images')
parser.add_argument('--checkpoint_dir', default='./checkpoints',
                    help='folder to output checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train',
                    help='model mode. default=`train`, option=`generate`')

opt = parser.parse_args()

try:
  os.makedirs(opt.outf)
  os.makedirs("unknown")
  os.makedirs(opt.checkpoint_dir)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


def train():
  """ train model
  """
  dataset = dset.MNIST(root=opt.dataroot,
                       download=True,
                       train=True,
                       transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(
                           (0.5,), (0.5,)),
                       ]))

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator(ngpu))
    netD = torch.nn.DataParallel(Discriminator(ngpu))
  else:
    netG = Generator(ngpu)
    netD = Discriminator(ngpu)
  netD.to(device)
  netG.to(device)
  if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))
  print(netG)
  print(netD)

  optimizerD = RMSprop(netD.parameters(), lr=opt.lr)
  optimizerG = RMSprop(netG.parameters(), lr=opt.lr)

  fixed_noise = torch.randn(opt.batch_size, opt.nz, device=device)

  print("########################################")
  print(f"Train dataset path: {opt.dataroot}")
  print(f"Batch size: {opt.batch_size}")
  print(f"Image size: {opt.img_size}")
  print(f"Epochs: {opt.epochs}")
  print("########################################")
  print("Starting trainning!")
  for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader):
      # get data
      real_imgs = data[0].to(device)
      batch_size = real_imgs.size(0)

      # Sample noise as generator input
      noise = torch.randn(batch_size, 100, device=device)

      ##############################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################

      optimizerD.zero_grad()

      # Generate a batch of images
      fake_imgs = netG(noise).detach()

      # Adversarial loss
      errD = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))

      errD.backward()
      optimizerD.step()

      # Clip weights of discriminator
      for p in netD.parameters():
        p.data.clamp_(-opt.clip_value, opt.clip_value)

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      if i % opt.n_critic == 0:
        optimizerG.zero_grad()

        # Generate a batch of images
        fake_imgs = netG(noise)

        # Adversarial loss
        errG = -torch.mean(netD(fake_imgs))

        errG.backward()
        optimizerG.step()
        print(f"Epoch->[{epoch + 1:3d}/{opt.epochs}] "
              f"Progress->[{i}/{len(dataloader)}] "
              f"Loss_D: {errD.item():.4f} "
              f"Loss_G: {errG.item():.4f} ", end="\r")

      if i % 5 == 0:
        vutils.save_image(
          real_imgs, f"{opt.outf}/mnist_real_samples.png", normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(
        ), f"{opt.outf}/mnist_fake_samples_epoch_{epoch + 1}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.checkpoint_dir}/mnist_G.pth")
    torch.save(netD.state_dict(), f"{opt.checkpoint_dir}/mnist_D.pth")


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator(ngpu))
  else:
    netG = Generator(ngpu)
  netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  netG.to(device)
  print(f"Load mnist model successful!")
  with torch.no_grad():
    for i in range(64):
      z = torch.randn(64, opt.nz, device=device)
      vutils.save_image(
        netG(z).detach(), f"unknown/mnist_fake_{i + 1:04d}.png", normalize=True)
  print("Images have been generated!")


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)
