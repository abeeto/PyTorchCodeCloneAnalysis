import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from config import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.batchnorm3 = nn.BatchNorm1d(256*2)
        self.fc4 = nn.Linear(512, 1024)
        self.batchnorm4 = nn.BatchNorm1d(256*4)
        self.fc5 = nn.Linear(1024, 28*28)

    def forward(self, x):
        x = self.batchnorm1(F.leaky_relu(self.fc1(x), 0.2))
        x = self.batchnorm2(F.leaky_relu(self.fc2(x), 0.2))
        x = self.batchnorm3(F.leaky_relu(self.fc3(x), 0.2))
        x = self.batchnorm4(F.leaky_relu(self.fc4(x), 0.2))
        x = torch.sigmoid(self.fc5(x))
        x = x.view(-1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(mini_batch_num, 28*28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x

class GAN():
    def __init__(self, dataloader, interrupting=False, eval=False):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.g_opt = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        if interrupting:
            self.discriminator.load_state_dict(torch.load("./param/tmp_d_weight"))
            self.generator.load_state_dict(torch.load("./param/tmp_g_weight"))
            self.d_opt.load_state_dict(torch.load("./param/tmp_d_opt_weight"))
            self.g_opt.load_state_dict(torch.load("./param/tmp_g_opt_weight"))
        if eval:
            self.generator.load_state_dict(torch.load(g_weight_dir))
            self.discriminator.load_state_dict(torch.load(d_weight_dir))

        self.criterion = nn.BCELoss()
        self.dataloader = dataloader
        self.d_loss_transition = []
        self.g_loss_transition = []
        self.max_acc = 0

    def study(self, epoch):
        g_loss_total = 0
        d_loss_total = 0
        valid_acc = 0
        valid_repeat = 0

        for repeat, (real_data, _) in enumerate(self.dataloader, 1):
            if repeat*mini_batch_num > 4000:
                continue
            # train data
            if repeat * mini_batch_num < data_num*(1-valid_rate):
                # renew discriminator
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

                # generatorの更新
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
                # 1000ごとに表示
                if mini_batch_num*repeat%10000 == 0:
                    print("   step[%d/%d]:"%(repeat*mini_batch_num, data_num))
                    print("        d loss %lf"%(d_loss))
                    print("        g loss %lf"%(g_loss))

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

        # if valid acc improve
        if valid_acc/valid_repeat > self.max_acc:
            self.max_acc = valid_acc/valid_repeat
            torch.save(self.discriminator.state_dict(), d_weight_dir)
            torch.save(self.generator.state_dict(), g_weight_dir)

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        eval_num = 1000
        real_acc = 0    # if real data is recognize as correct by discriminator
        fake_acc = 0    # if generated data is recognize as fake by discriminator

        for repeat, (real_data, _) in enumerate(self.dataloader, 1):
            if mini_batch_num*repeat > eval_num:
                continue
            # real
            real_correct = np.ones(mini_batch_num)
            real_pred = self.discriminator(real_data).view(mini_batch_num)
            real_pred = torch.round(real_pred)
            real_acc += (torch.Tensor(real_correct) == real_pred).sum().item()
            # fake
            fake_correct = np.zeros(mini_batch_num)
            input_noise = self.noise_generator()
            fake_data = self.generator(input_noise)
            fake_pred = self.discriminator(fake_data).view(mini_batch_num)
            fake_pred = torch.round(fake_pred)
            fake_acc += (torch.Tensor(fake_correct) == fake_pred).sum().item()
        print("Discriminator Acc:\n      correct acc: %lf\n      fake acc: %lf\n"
                %(real_acc/eval_num, fake_acc/eval_num))

    def eval_pic(self, epoch):
        self.generator.eval()
        concat_pic = np.array([])
        for col in range(1, pic_size+1):
            tmp = np.array([])
            for row in range(1, pic_size+1):
                #pic = np.array(mnist_data[int(row+col*pic_size)][0]).reshape(28, 28)
                input_noise = self.noise_generator()[0].view(1, noise_dim)
                pic = self.generator(input_noise).detach().numpy()
                pic = pic.reshape(28, 28)
                #plt.figure()
                #plt.imsave("./result/%d_only.png"%(epoch), pic)

                if row == 1:
                    tmp = pic
                else:
                    tmp = np.concatenate([tmp, pic], axis=1)
            if col == 1:
                concat_pic = tmp
            else:
                concat_pic = np.concatenate([concat_pic, tmp], axis=0)
        plt.figure()
        plt.imsave("./result/%d.png"%(epoch), concat_pic)
        plt.close()
        del concat_pic
        del pic

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

    def output(self, epoch):
        plt.figure()
        plt.plot(range(epoch), self.g_loss_transition, label="generator loss")
        plt.plot(range(epoch), self.d_loss_transition, label="discriminator loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("./result/loss_transition.png")
        plt.close()
