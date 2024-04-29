import utils
import dcgan
import waae_resnet
from config import args
import os
import numpy as np
import itertools
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch
import scipy.io
import math

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class WAAE():

    def __init__(self):

        # load images and scale in range (-1, 1)
        self.images_loader = utils.load_images('datasets/patterns/')
        # dataset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
        # dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20, drop_last=True)

        # load networks
        if args.architecture == 'ResNet':
            self.encoder = utils.load_network('encoder', waae_resnet.Encoder())
            self.decoder = utils.load_network('decoder', waae_resnet.Decoder())
            self.discriminator = utils.load_network('discriminator', waae_resnet.Discriminator())

        # print total and trainable parameters for networks
        utils.print_parameters(self.encoder, 'Encoder')
        utils.print_parameters(self.decoder, 'Decoder')
        utils.print_parameters(self.discriminator, 'Discriminator')

        self.reconstruct_loss = torch.nn.MSELoss().cuda() if cuda else torch.nn.MSELoss()

        # set up optimizers
        self.optimizer_R = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.lr_R, betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_D, betas=(args.b1, args.b2))
        #self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=args.lr_D, momentum=0.9, dampening=0, weight_decay=1e-4)
        self.optimizer_G = torch.optim.Adam(self.encoder.parameters(), lr=args.lr_G, betas=(args.b1, args.b2))

        # set up LR schedulers
        self.scheduler_R = StepLR(self.optimizer_R, step_size=args.lr_step, gamma=args.lr_gamma)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
        self.scheduler_G = StepLR(self.optimizer_G, step_size=args.lr_step, gamma=args.lr_gamma)

        # create batch of latent vectors to visualize the progression of the generator
        self.fixed_noise = Variable(Tensor(np.random.normal(0, args.s_sd, (100, args.latent_dim))))

    def train(self):

        r_total_loss = []
        d_total_loss = []
        g_total_loss = []

        for epoch in range(args.n_epochs):

            r_epoch_loss = 0
            d_epoch_loss = 0
            g_epoch_loss = 0

            for batch, (imgs, _) in enumerate(self.images_loader):

                with torch.no_grad():
                    imgs = Variable(imgs.type(Tensor))

                #  Train Autoencoder - Reconstruction

                self.optimizer_R.zero_grad()
                if args.architecture == 'ResNet':
                    encoded_imgs = self.encoder(imgs)
                    decoded_imgs = self.decoder(encoded_imgs)
                r_loss = self.reconstruct_loss(decoded_imgs, imgs)
                r_loss.backward(retain_graph=True)
                self.optimizer_R.step()

                #  Train Discriminator

                for k in range(args.k):

                    self.optimizer_D.zero_grad()
                    with torch.no_grad():
                        # sample noise
                        z = Variable(Tensor(np.random.normal(0, args.s_sd, (imgs.shape[0], args.latent_dim))))

                    # measure discriminator's ability to classify real from generated samples
                    d_real = self.discriminator(utils.gaussian(z, 0, args.n_sd))
                    d_fake = self.discriminator(utils.gaussian(encoded_imgs.detach(), 0, args.n_sd))
                    d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                    d_loss.backward()
                    self.optimizer_D.step()

                    # clip discriminator's weights
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-args.clip_value, args.clip_value)

                # Train Generator

                self.optimizer_G.zero_grad()
                d_fake = self.discriminator(utils.gaussian(encoded_imgs, 0, args.n_sd))
                g_loss = -torch.mean(d_fake)
                g_loss.backward()
                self.optimizer_G.step()

                # write losses to files
                with torch.no_grad():
                    utils.save_loss(epoch+1, batch+1, len(self.images_loader), r_loss.item(), 'reconstruction')
                    utils.save_loss(epoch+1, batch+1, len(self.images_loader), d_loss.item(), 'discriminator')
                    utils.save_loss(epoch+1, batch+1, len(self.images_loader), g_loss.item(), 'generator')

                    r_epoch_loss += r_loss.item()
                    d_epoch_loss += d_loss.item()
                    g_epoch_loss += g_loss.item()

                    batches_done = epoch * len(self.images_loader) + batch
                    if batches_done % args.sample_interval == 0:
                        utils.save_images(imgs, 'real', n_row=int(math.sqrt(args.batch_size)), batches_done=batches_done)
                        utils.save_images(decoded_imgs.detach(), 'recon', n_row=int(math.sqrt(args.batch_size)), batches_done=batches_done)
                        print('Images and their reconstructions saved')

            with torch.no_grad():
                r_total_loss.append(r_epoch_loss/len(self.images_loader))
                d_total_loss.append(d_epoch_loss/len(self.images_loader))
                g_total_loss.append(g_epoch_loss/len(self.images_loader))

            # save loss plots
            utils.save_plot(epoch, r_total_loss, 'Reconstruction')
            utils.save_plot(epoch, d_total_loss, 'Discriminator')
            utils.save_plot(epoch, g_total_loss, 'Generator')
            print('Plots saved')

            # save images generated from fixed noise
            gen_random = self.decoder(self.fixed_noise)
            utils.save_images(gen_random.detach(), 'generated_fixed', n_row=10, batches_done=batches_done)
            print('Images generated from fixed noise saved')

            if epoch % 5 == 0:
                # save images generated from random noise
                z = Variable(Tensor(np.random.normal(0, args.s_sd, (100, args.latent_dim))))
                gen_random = self.decoder(z)
                utils.save_images(gen_random.detach(), 'generated_random', n_row=10, batches_done=batches_done)
                print('Images generated from random noise saved')
                #utils.save_manifold(batches_done, self.decoder)
                #print('Manifold saved')

                # save models
                torch.save(self.encoder.state_dict(), '{}/encoder.pth'.format(args.folder_name))
                torch.save(self.decoder.state_dict(), '{}/decoder.pth'.format(args.folder_name))
                torch.save(self.discriminator.state_dict(), '{}/discriminator.pth'.format(args.folder_name))
                print('Models saved')

            # decay learning rate
            self.scheduler_R.step()
            print('Epoch:', epoch, 'R LR:', self.scheduler_R.get_lr())
            self.scheduler_D.step()
            print('Epoch:', epoch, 'D LR:', self.scheduler_D.get_lr())
            self.scheduler_G.step()
            print('Epoch:', epoch, 'G LR:', self.scheduler_G.get_lr())


class DCGAN():
    def __init__(self):

        # load images and scale in range (-1, 1)
        self.images_loader = utils.load_images('datasets/patterns/')
        # dataset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
        # dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20, drop_last=True)

        # Define the network
        self.generator = utils.load_network('generator', dcgan.Generator())
        self.discriminator = utils.load_network('discriminator', dcgan.Discriminator())

        # print total and trainable parameters for networks
        utils.print_parameters(self.generator, 'Generator')
        utils.print_parameters(self.discriminator, 'Discriminator')

        self.adversarial_loss = torch.nn.BCELoss().cuda() if cuda else torch.nn.BCELoss()

        # set up optimisers
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=args.lr_d)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=args.lr_g)

        # set up LR schedulers
        self.scheduler_D = StepLR(self.optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
        self.scheduler_G = StepLR(self.optimizer_G, step_size=args.lr_step, gamma=args.lr_gamma)

        # create latent vectors to visualize the progression of the generator
        self.fixed_noise = Variable(Tensor(np.random.normal(0, args.s_sd, (args.batch_size, args.latent_dim))))

    def train(self):

        g_total_loss = []
        d_total_loss = []

        for epoch in range(args.epochs):

            g_epoch_loss = 0
            d_epoch_loss = 0

            for batch, (imgs, _) in enumerate(self.images_loader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(self.real_label), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(self.fake_label), requires_grad=False)

                with torch.no_grad():
                    real_imgs = Variable(imgs.type(Tensor))

                # Train Discriminator

                self.optimizer_D.zero_grad()
                z = torch.randn(imgs.shape[0], args.nz, 1, 1)
                # generate fake images from latent vectors
                gen_imgs = self.generator(z)

                # train on real data
                d_real = self.discriminator(utils.gaussian(real_imgs, 0, args.n_sd))
                real_loss = self.adversarial_loss(d_real, args.real)
                real_loss.backward()

                # train on fake data
                d_fake = self.discriminator(utils.gaussian(gen_imgs.detach(), 0, args.n_sd))
                fake_loss = self.adversarial_loss(d_fake, args.fake)
                fake_loss.backward()

                self.optimizer_D.step()

                d_loss = real_loss + fake_loss
                # d_loss.backward()

                # Train Generator

                self.optimizer_G.zero_grad()

                gen_imgs = self.generator(z)
                d_gen = self.discriminator(utils.gaussian(gen_imgs, 0, args.n_sd))
                g_loss = self.adversarial_loss(d_gen, args.real)

                g_loss.backward()
                self.optimizer_G.step()

                # write losses to files
                with torch.no_grad():
                    utils.save_loss(epoch+1, batch+1, len(self.images_loader), d_loss.item(), 'discriminator')
                    utils.save_loss(epoch+1, batch+1, len(self.images_loader), g_loss.item(), 'generator')

                    # Save Losses for plotting later
                    g_epoch_loss += g_loss.item()
                    d_epoch_loss += d_loss.item()

                    batches_done = epoch * len(self.images_loader) + batch
                    if batches_done % args.sample_interval == 0:
                        with torch.no_grad():
                            generated = self.generator(self.fixed_noise)
                        utils.save_images(generated.detach(), 'generated', n_row=8, batches_done=batches_done)
                        print("Generated samples saved")

            g_total_loss.append(g_epoch_loss / len(self.images_loader))
            d_total_loss.append(d_epoch_loss / len(self.images_loader))

            utils.save_plot(epoch + 1, g_total_loss, "Generator")
            utils.save_plot(epoch + 1, d_total_loss, "Discriminator")
            print("Plots saved")

            # decay learning rate
            self.scheduler_D.step()
            print('Epoch:', epoch, 'D LR:', self.scheduler_D.get_lr())
            self.scheduler_G.step()
            print('Epoch:', epoch, 'G LR:', self.scheduler_G.get_lr())
