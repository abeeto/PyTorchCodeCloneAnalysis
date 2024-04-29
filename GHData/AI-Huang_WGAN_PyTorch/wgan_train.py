#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-21 13:57
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from gan.wgan import Generator, Discriminator
from dataloader import get_dataloader


def parse_cmd_args():
    # Experiment settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['svhn', 'cifar10'], help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--lr", type=float,
                        default=0.00005, help="learning rate")

    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=0,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=0,
                        help="number of image channels")

    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int,
                        default=400, help="interval betwen image samples")

    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        if args.data_root == None:
            args.data_root = "~/.datasets"
        args.img_size = 32
        args.channels = 3
    elif args.dataset == 'svhn':
        if args.data_root == None:
            args.data_root = "~/.datasets/SVHN"
        args.img_size = 32
        args.channels = 3

    return args


def train(dataloader, discriminator, generator, optimizer_D, optimizer_G, use_cuda, args):
    """train
    The training function of WGAN
    """
    os.makedirs("images", exist_ok=True)
    # args.log_dir
    with open(os.path.join(".", "loss.csv"), "w") as f:
        f.write("Epoch,Batch,D loss,G loss\n")

    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    batches_done = 0
    for epoch in range(args.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # Train Discriminator
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()

            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + \
                torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # Train the generator every n_critic iterations
            if i % args.n_critic == 0:
                # Train Generator
                optimizer_G.zero_grad()
                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )
                # args.log_dir
                with open(os.path.join(".", "loss.csv"), "a") as f:
                    f.write("%d,%d,%f,%f\n" %
                            (epoch, batches_done, loss_D.item(), loss_G.item()))

            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" %
                           batches_done, nrow=5, normalize=True)
            batches_done += 1


def main():
    args = parse_cmd_args()
    print(args)
    input_shape = (args.channels, args.img_size, args.img_size)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Initialize generator and discriminator
    generator = Generator(input_shape=input_shape, latent_dim=args.latent_dim)
    discriminator = Discriminator(input_shape=input_shape)

    if use_cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    train_loader, _ = get_dataloader(args)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    train(train_loader, discriminator=discriminator, generator=generator,
          optimizer_D=optimizer_D, optimizer_G=optimizer_G, use_cuda=use_cuda, args=args)


if __name__ == "__main__":
    main()
