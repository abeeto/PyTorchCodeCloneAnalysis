from __future__ import print_function
from pathlib import Path
import torchvision.transforms as T
import torch
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
import numpy as np
from dataset import UpsampleDataset
from model import Model
from tqdm import tqdm
from skimage import io, transform
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb

from model2 import Discriminator, Model2
from model3 import Model3


workers = 0
nc = 3
nz = 25


lr = 1e-4
weight_decay = 1e-6

ngpu = 1

from dotenv import load_dotenv

NO_WANDB = False




def train(i_image_size, o_image_size, epochs, dataroot, batch_size, checkpoints, inplace_dataset):
    global NO_WANDB


    last_mean_loss = 10000000
    if 'NO_WANDB' in os.environ:
        NO_WANDB = True
    else:
        wandb.init(project="upscaling")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using device {device}')


    # Models
    netG = Model3().to(device)
    netD = Discriminator().to(device)

    # if device.type == "cuda":
    #     netG = netG.half()
    #     netD = netD.half()

    # Optimizers

    beta1 = 0.5
    # if device.type == "cuda":
    #     import bitsandbytes as bnb
    #     optimizerD = bnb.optim.Adam8bit(netD.parameters(), lr=lr) # add bnb optimizer
    #     optimizerG = bnb.optim.Adam8bit(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # add bnb optimizer
    # else:
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Loss functions
    beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss

    content_loss_criterion = nn.MSELoss() # for pixel difference loss
    adversarial_loss_criterion = nn.BCEWithLogitsLoss() # for discriminator loss

    print(f'{sum(p.numel() for p in netG.parameters()):,} parameters')
    dataset = UpsampleDataset(dataroot, i_image_size, o_image_size, is_inplace=inplace_dataset)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    accumulation_steps = 1 # we want to do model's update only after 64 images being processed


    for epoch in tqdm(range(0, epochs)):
        losses = np.array([])
        for batch_idx, data in enumerate(dataloader, 0):
            
            
            lr_img = data[0].to(device, dtype=torch.float)
            hr_img = data[1].to(device, dtype=torch.float)

            # if device.type == "cuda":
            #     lr_img = lr_img.half()
            #     hr_img = hr_img.half()
            
            # GENERATOR UPDATE

            # Generate
            sr_img = netG(lr_img)

            # Discriminate super-resolved (SR) images
            sr_discriminated = netD(hr_img)
            
            
            content_loss = content_loss_criterion(sr_img, hr_img)
            adversarial_loss_g = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            perceptual_loss = content_loss + beta * adversarial_loss_g
            
            # Back-prop.
            
            losses = np.append(losses, perceptual_loss.item())

            perceptual_loss = perceptual_loss / accumulation_steps
            perceptual_loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # Update generator
                optimizerG.step()
                optimizerG.zero_grad()

            # DISCRIMINATOR UPDATE

            # Discriminate super-resolution (SR) and high-resolution (HR) images
            hr_discriminated = netD(hr_img)
            sr_discriminated = netD(sr_img.detach())

            # Binary Cross-Entropy loss
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

            adversarial_loss = adversarial_loss / accumulation_steps
            adversarial_loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # Update discriminator
                optimizerD.step()
                optimizerD.zero_grad()
            
            if not NO_WANDB:
                # NO_WANDB=true

                wandb.log({ 'content_loss': content_loss.item(), 'adversarial_loss_d': adversarial_loss.item(), 'adversarial_loss_g': adversarial_loss_g.item(), 'loss': perceptual_loss.item()   })

                if batch_idx % 100 == 0:
                    slides = torch.cat((
                        T.Resize((o_image_size, o_image_size))(lr_img[0:8]),
                        hr_img[0:8],
                        sr_img[0:8]))
                    samples = wandb.Image(slides, caption="Upscaled")
                    wandb.log({ 'samples': samples})
        print(f'Epoch {epoch}, Mean Loss: {np.mean(losses)}')

        if checkpoints and last_mean_loss > np.mean(losses):
            last_mean_loss = np.mean(losses)
            torch.save(netG.state_dict(), f'./checkpoints/best_g.pth')
            torch.save(netD.state_dict(), f'./checkpoints/best_d.pth')


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-nocp', action='store_true', dest='no_checkpoint')
    parser.add_argument('-inplace', action='store_true', dest='inplace_dataset')
    parser.add_argument('-data', action='store', dest='path')
    parser.add_argument('-batch', action='store', dest='batch_size', type=int)
    parser.add_argument('-epochs', action='store', dest='epochs', type=int)
    parser.add_argument('-i_size', action='store', dest='i_size', type=int)
    parser.add_argument('-o_size', action='store', dest='o_size', type=int)

    args = parser.parse_args()

    print(f'Using dataset {args.path}')
    print(f'Using batch size {args.batch_size}')
    print(f'Saving checkpoints: {not args.no_checkpoint}')
    print(f'Inplace dataset: {args.inplace_dataset}')

    print(f'Upscaling from {args.i_size}x{args.i_size} to {args.o_size}x{args.o_size}')

    print(f'Training for {args.epochs} epochs')
    Path('./checkpoints').mkdir(exist_ok=True)

    train(args.i_size, args.o_size, args.epochs, args.path, args.batch_size, not args.no_checkpoint, args.inplace_dataset)