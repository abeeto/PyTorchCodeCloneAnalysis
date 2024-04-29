#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-03-21 23:49
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import argparse
import os
import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image

from gan.wgan_gp_resnet import Generator, Discriminator
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
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_gp", type=float, default=10,
                        help="lambda_gp: loss weight for gradient penalty")

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
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Train
    train(train_loader, discriminator, generator,
          optimizer_D, optimizer_G, use_cuda, args)


def compute_gradient_penalty(D, real_samples, fake_samples, use_cuda):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(dataloader, discriminator, generator, optimizer_D, optimizer_G, use_cuda, args):
    """train
    The training function of WGAN GP
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
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data, use_cuda=use_cuda)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + \
                torch.mean(fake_validity) + args.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                # Train Generator
                # optimizer_G.zero_grad()
                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # args.log_dir
                with open(os.path.join(".", "loss.csv"), "a") as f:
                    f.write("%d,%d,%f,%f\n" %
                            (epoch, batches_done, d_loss.item(), g_loss.item()))

                if batches_done % args.sample_interval == 0:
                    save_image(fake_imgs.data[:25], "images/%d.png" %
                               batches_done, nrow=5, normalize=True)

                batches_done += args.n_critic


if __name__ == "__main__":
    main()
