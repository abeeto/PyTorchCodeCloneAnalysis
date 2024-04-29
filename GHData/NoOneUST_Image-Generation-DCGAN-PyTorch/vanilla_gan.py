#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This is the main training file for the vanilla GAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from vanilla_utils import create_dir, create_model, checkpoint, sample_noise, save_samples

# draw loss gragh
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

SEED = 11
num_workers = 4
var_label = 0.1

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(train_loader, opts, device):
    
    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)

    if device.type == 'cuda' and num_workers > 0:
        D = nn.DataParallel(D, list(range(num_workers)))
        G = nn.DataParallel(G, list(range(num_workers)))
    
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    
    fixed_noise = sample_noise(opts.batch_size, opts.noise_size).to(device)
    
    iteration = 1
    
    mse_loss = torch.nn.MSELoss()
#     bce_loss = torch.nn.BCELoss()
    total_train_iters = opts.num_epochs * len(train_loader)
    
    d_real_loss = []
    d_fake_loss = []
    g_loss = []
    for epoch in range(opts.num_epochs):

        for batch in train_loader:

            real_images = batch[0].to(device)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            # D_real_loss = ...
            real_output = D(real_images).reshape((-1, 1)).to(device)
            var_Value = var_label * torch.ones(opts.batch_size, 1).to(device)
            refSeq = 1 - torch.trunc(1.1 * torch.rand(opts.batch_size, 1)).to(device)
            real_labels = torch.normal(mean=1, std=var_Value).mul(refSeq)
            D_real_loss = mse_loss(real_output, real_labels)

            # 2. Sample noise
            # noise = ...
            noise = sample_noise(opts.batch_size, opts.noise_size).to(device)

            # 3. Generate fake images from the noise
            # fake_images = ...
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            # D_fake_loss = ...
            fake_output = D(fake_images).reshape((-1, 1))
            fake_labels = torch.zeros(opts.batch_size,1).to(device)
            D_fake_loss = mse_loss(fake_output,fake_labels)

            # 5. Compute the total discriminator loss
            # D_total_loss = ...
            D_total_loss = D_real_loss/2 + D_fake_loss/2
    
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            # noise = ...
            noise = sample_noise(opts.batch_size, opts.noise_size).to(device)

            # 2. Generate fake images from the noise
            # fake_images = ...
            fake_images = G(noise)
            
            # 3. Compute the generator loss
            # G_loss = ...
            fake_output = D(fake_images).reshape((-1, 1))
            G_loss = mse_loss(fake_output,real_labels)

            G_loss.backward()
            g_optimizer.step()


            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
                d_real_loss.append(D_real_loss.item())
                d_fake_loss.append(D_fake_loss.item())
                g_loss.append(G_loss.item())
            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1
        
    epochs = range(0,len(d_real_loss)*opts.log_step, opts.log_step)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(epochs,d_real_loss,'r',label='D_real_loss')
    plt.plot(epochs,d_fake_loss,'b',label='D_fake_loss')
    plt.plot(epochs,g_loss,'y',label='G_loss')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("Vanilla-GAN.png")
    
    
    
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    train_loader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train(train_loader, opts, device)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)
