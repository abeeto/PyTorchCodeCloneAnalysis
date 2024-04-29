import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from config import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channel_size = 64
        self.fc = nn.Linear(noise_dim, channel_size*36)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=channel_size*4, out_channels=channel_size*2,
            kernel_size=3, stride=2
            )
        self.batchnorm2 = nn.BatchNorm2d(channel_size*2)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=channel_size*2, out_channels=channel_size,
            kernel_size=2, stride=2
            )
        self.batchnorm3 = nn.BatchNorm2d(channel_size)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=channel_size, out_channels=1,
            kernel_size=2, stride=2
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, 3, 3)
        x = self.batchnorm2(F.relu(self.deconv1(x)))
        x = self.batchnorm3(F.relu(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        channel_size = 64
        self.conv1 = nn.Conv2d(
            in_channels = 1, out_channels = channel_size,
            kernel_size = 2, stride = 2, padding=0
            )
        self.batchnorm1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(
            in_channels = channel_size, out_channels = channel_size*2,
            kernel_size = 2, stride = 2, padding=0
            )
        self.batchnorm2 = nn.BatchNorm2d(channel_size*2)

        self.conv3 = nn.Conv2d(
            in_channels = channel_size*2, out_channels = channel_size*4,
            kernel_size = 3, stride = 2, padding=0
            )
        self.batchnorm3 = nn.BatchNorm2d(channel_size*4)

        self.conv4 = nn.Conv2d(
            in_channels = channel_size*4, out_channels = 1,
            kernel_size = 3, stride = 2, padding=0
            )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.sigmoid(self.conv4(x))
        x = x.view(-1, 1)
        return x

class GAN():
    def __init__(self, dataloader, interrupting=False, eval=False):
        self.discriminator = Discriminator()
        self.generator = Generator()

        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_opt = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.real_acc_transition = []
        self.fake_acc_transition = []

        if interrupting:
            discriminator.load_state_dict(torch.load("./param/tmp_d_weight"))
            generator.load_state_dict(torch.load("./param/tmp_g_weight"))
            d_opt.load_state_dict(torch.load("./param/tmp_d_opt_weight"))
            g_opt.load_state_dict(torch.load("./param/tmp_g_opt_weight"))
        if eval:
            generator.load_state_dict(torch.load(g_weight_dir))
            discriminator.load_state_dict(torch.load(d_weight_dir))

        self.criterion = nn.MSELoss()
        self.dataloader = dataloader
        self.d_loss_transition = []
        self.g_loss_transition = []
        self.max_acc = 0

    def study(self, epoch):
        g_loss_total = 0
        d_loss_total = 0
        valid_acc = 0
        valid_repeat = 0
        display_step = 1000
        display_step_times = 1

        for repeat, (real_data, _) in enumerate(self.dataloader, 1):
            if repeat*mini_batch_num > 5000:
                continue

            # train data
            if repeat * mini_batch_num < data_num*(1-valid_rate):
                # discriminator
                self.discriminator.train()
                self.generator.eval()
                self.d_opt.zero_grad()
                # real
                real_correct = np.ones((mini_batch_num, 1),dtype=np.float)
                real_correct = torch.Tensor(real_correct)
                real_data = Variable(real_data)
                real_pred = self.discriminator(real_data)
                real_loss = self.criterion(real_pred, real_correct)
                # fake
                input_noise = self.noise_generator(False)
                fake_data = self.generator(input_noise)
                fake_correct = np.zeros((mini_batch_num, 1),dtype=np.float)
                fake_correct = torch.Tensor(fake_correct)
                fake_pred = self.discriminator(Variable(fake_data.detach()))
                fake_loss = self.criterion(fake_pred, fake_correct)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                self.d_opt.step()
                d_loss_total += d_loss.item()

                # generator
                self.discriminator.eval()
                self.generator.train()
                self.g_opt.zero_grad()
                input_noise = self.noise_generator()
                correct = np.ones((mini_batch_num, 1),dtype=np.float)
                correct = torch.Tensor(correct)
                data = self.generator(input_noise)
                pred = self.discriminator(data)
                g_loss = self.criterion(pred, correct)
                g_loss.backward()
                self.g_opt.step()
                g_loss_total += g_loss.item()

        # valid data
        else:
                self.discriminator.eval()
                self.generator.eval()

                input_noise = self.noise_generator()
                fake_data = self.generator(input_noise)
                fake_pred = self.discriminator(fake_data).view(mini_batch_num)
                fake_pred = fake_pred.detach().numpy()
                valid_acc += np.average(fake_pred)
                valid_repeat += 1

        self.d_loss_transition.append(d_loss_total)
        self.g_loss_transition.append(g_loss_total)

        if valid_acc/valid_repeat > self.max_acc:
            self.max_acc = valid_acc/valid_repeat
            torch.save(self.discriminator.state_dict(), d_weight_dir)
            torch.save(self.generator.state_dict(), g_weight_dir)

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        eval_num = 1000
        real_acc = 0
        fake_acc = 0

        for repeat, (real_data, _) in enumerate(self.dataloader, 1):
            if mini_batch_num*repeat > eval_num:
                continue
            # real
            real_correct = np.ones(mini_batch_num)
            real_pred = self.discriminator(real_data).view(mini_batch_num)
            real_acc += real_pred.sum().item()
            # fake
            fake_correct = np.zeros(mini_batch_num)
            input_noise = self.noise_generator()
            fake_data = self.generator(input_noise)
            fake_pred = self.discriminator(fake_data).view(mini_batch_num)
            fake_acc += fake_pred.sum().item()
        print("Discriminator Acc:\n      correct acc: %lf\n      fake acc: %lf\n"
                %(real_acc/eval_num, fake_acc/eval_num))
        self.real_acc_transition.append(real_acc/eval_num)
        self.fake_acc_transition.append(fake_acc/eval_num)

    def eval_pic(self, epoch):
        self.generator.eval()
        concat_pic = np.array([])
        for col in range(1, pic_size+1):
            tmp = np.array([])
            for row in range(1, pic_size+1):
                input_noise = self.noise_generator()[0].view(1, noise_dim)
                pic = self.generator(input_noise).detach().numpy()
                pic = pic.reshape(28, 28)

                if row == 1:
                    tmp = pic
                else:
                    tmp = np.concatenate([tmp, pic], axis=1)
            if col == 1:
                concat_pic = tmp
            else:
                concat_pic = np.concatenate([concat_pic, tmp], axis=0)
            del tmp
            del input_noise
        plt.figure()
        plt.imsave("./result/%d.png"%(epoch), concat_pic)
        plt.close()
        del concat_pic

    def noise_generator(self, is_training=True):
        if is_training:
            return Variable(
                    torch.Tensor(
                        np.random.randn(mini_batch_num, noise_dim)
                        )
                    )
        else:
            return torch.Tensor(
                    np.random.randn(mini_batch_num, noise_dim)
                    )

    def save_tmp_weight(self, epoch):
        torch.save(self.discriminator.state_dict(), "./param/tmp_d_weight")
        torch.save(self.generator.state_dict(), "./param/tmp_g_weight")
        torch.save(self.d_opt.state_dict(), "./param/tmp_d_opt_weight")
        torch.save(self.g_opt.state_dict(), "./param/tmp_g_opt_weight")
        f = open("./param/tmp.pickle", mode="wb")
        pickle.dump(epoch, f)

    """
    def output(self, epoch=0):
        plt.figure()
        plt.plot(range(epoch), self.real_acc_transition, label="real acc")
        plt.plot(range(epoch), self.fake_acc_transition, label="fake acc")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.savefig("./result/acc_transition.png")
        plt.close()
    """
