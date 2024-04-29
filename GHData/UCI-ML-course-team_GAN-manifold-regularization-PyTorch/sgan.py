import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter


class SGAN_Manifold_Reg():
    def __init__(self, batch_size, latent_dim, num_classes, generator, discriminator, data_loaders):

        self.batch_size = batch_size
        self.batch_size_cuda = torch.tensor(self.batch_size).cuda()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lr = 0.0003

        self.save_path = 'models/'
        self.log_dir = 'logs/'
        self.writer = SummaryWriter(self.log_dir)

        self.G = generator.cuda()
        self.D = discriminator.cuda()
        self.train_unl_loader, self.train_lb_loader, self.valid_loader, self.test_loader = data_loaders

        self.ce_criterion = nn.CrossEntropyLoss().cuda()
        self.mse = nn.MSELoss().cuda()

        self.train_step = 0
        self.test_step = 0

    def train(self, num_epochs):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        for epoch_idx in range(num_epochs):

            avg_G_loss = avg_D_loss = 0
            self.G.train()
            self.D.train()
            for unl_train_x, __ in self.train_unl_loader:
                lb_train_x, lb_train_y = next(self.train_lb_loader)
                unl_train_x = unl_train_x.cuda()
                lb_train_x = lb_train_x.cuda()
                lb_train_y = lb_train_y.cuda()

                z, z_perturbed = self.define_noise()

                # Train Discriminator
                opt_D.zero_grad()
                imgs_fake = self.G(z)
                imgs_fake_perturbed = self.G(z_perturbed)

                __, logits_lb = self.D(lb_train_x)
                features_fake, logits_fake = self.D(imgs_fake)
                features_fake_pertubed, __ = self.D(imgs_fake_perturbed)
                features_real, logits_unl = self.D(unl_train_x)

                logits_sum_unl = torch.logsumexp(logits_unl, dim=1)
                logits_sum_fake = torch.logsumexp(logits_fake, dim=1)
                loss_unsupervised = torch.mean(F.softplus(logits_sum_unl)) - torch.mean(logits_sum_unl) + torch.mean(
                    F.softplus(logits_sum_fake))

                loss_supervised = torch.mean(self.ce_criterion(logits_lb, lb_train_y))
                loss_manifold_reg = self.mse(features_fake, features_fake_pertubed) \
                                    / self.batch_size_cuda

                loss_D = loss_supervised + .5 * loss_unsupervised + 1e-3 * loss_manifold_reg
                loss_D.backward()
                opt_D.step()
                avg_D_loss += loss_D

                # Train Generator
                opt_G.zero_grad()
                opt_D.zero_grad()
                imgs_fake = self.G(z)
                features_fake, __ = self.D(imgs_fake)
                features_real, __ = self.D(unl_train_x)
                m1 = torch.mean(features_real, dim=0)
                m2 = torch.mean(features_fake, dim=0)
                loss_G = torch.mean((m1 - m2) ** 2)  # Feature matching
                loss_G.backward()
                opt_G.step()
                avg_G_loss += loss_G

                self.writer.add_scalar('G_loss', loss_G, self.train_step)
                self.writer.add_scalar('D_loss', loss_D, self.train_step)
                self.train_step += 1

            # Evaluate
            avg_G_loss /= len(self.train_unl_loader)
            avg_D_loss /= len(self.train_unl_loader)

            acc, val_loss = self.eval()

            print('Epoch %d disc_loss %.3f gen_loss %.3f val_loss %.3f acc %.3f' % (
                epoch_idx, avg_D_loss, avg_G_loss, val_loss, acc))

            torch.save(self.D.state_dict(), self.save_path + 'disc_{}.pth'.format(epoch_idx))
            torch.save(self.G.state_dict(), self.save_path + 'gen_{}.pth'.format(epoch_idx))

        self.writer.close()

    def eval(self, test=False, epoch_idx=None):
        if test:
            self.D.load_state_dict(torch.load(self.save_path + 'disc_{}.pth'.format(epoch_idx)))
            eval_loader = self.test_loader
        else:
            eval_loader = self.valid_loader

        val_loss = corrects = total_samples = 0.0
        with torch.no_grad():
            self.D.eval()
            for x, y in eval_loader:
                x, y = x.cuda(), y.cuda()
                __, logits = self.D(x)
                loss = self.ce_criterion(logits, y)
                if not test:
                    self.writer.add_scalar('val_loss', loss, self.test_step)
                    self.test_step += 1
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                corrects += torch.sum(preds == y)
                total_samples += len(y)

            val_loss /= len(self.valid_loader)
            acc = corrects.item() / total_samples

        return acc, val_loss

    def define_noise(self):
        z = torch.randn(self.batch_size, self.latent_dim).cuda()
        z_perturbed = z + torch.randn(self.batch_size, self.latent_dim).cuda() * 1e-5
        return z, z_perturbed
