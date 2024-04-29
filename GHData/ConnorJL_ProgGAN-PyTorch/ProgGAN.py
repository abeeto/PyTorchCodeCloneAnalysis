import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.misc import imsave
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, calculate_gain


class ProgGAN(object):
    def __init__(self, nz=512, lr=0.0010):
        self.nz = nz  # Dimension of noise vector
        self.lr = lr

        self.current_size = 4
        self.batch_size = 16

        # Create Networks
        self.disc = Discriminator()
        self.gen = Generator()

        # Create Optimizers
        self.optimizerD = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0, 0.99))
        self.optimizerG = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0, 0.99))
        self.criterion = torch.nn.MSELoss()

        # Helper Tensors
        self.input = torch.FloatTensor(self.batch_size, 3, self.current_size, self.current_size)
        self.noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)
        self.label = torch.FloatTensor(self.batch_size)

    def cuda(self):
        self.disc.cuda()
        self.gen.cuda()
        self.criterion.cuda()
        self.input = self.input.cuda()
        self.noise = self.noise.cuda()
        self.label = self.label.cuda()

    def state_dict(self):
        state = {
            "disc": self.disc.state_dict(),
            "gen": self.gen.state_dict(),
            "optD": self.optimizerD.state_dict(),
            "optG": self.optimizerG.state_dict(),
            "alpha": self.disc.alpha,
            "current_size": self.current_size,
        }

        return state

    def load(self, state):
        while self.current_size < state["current_size"]:
            self.grow()
        self.disc.load_state_dict(state["disc"])
        self.gen.load_state_dict(state["gen"])
        self.optimizerD.load_state_dict(state["optD"])
        self.optimizerG.load_state_dict(state["optG"])
        self.disc.alpha = state["alpha"]
        self.gen.alpha = state["alpha"]

    def sample(self, path=None, size=9, torch_style=False):
        imgs = []
        while len(imgs) < size*size:
            self.noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
            noisev = Variable(self.noise)
            imgb = self.gen(noisev).data.cpu()
            if not torch_style:
                imgb = torch.transpose(imgb, 1, 3).numpy()
            for i in range(self.batch_size):
                imgs.append(imgb[i])
                if len(imgs) == size*size:
                    if path is not None and not torch_style:
                        self.color_grid_vis(imgs, size, size, save_path=path)
                    return imgs

    def color_grid_vis(self, X, nh, nw, save_path=None):
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
        for n, x in enumerate(X):
            j = int(n/nw)
            i = int(n%nw)
            img[j*h:j*h+h, i*w:i*w+w, :] = x
        if save_path is not None:
            imsave(save_path, img)
        return img

    def grow(self):
        self.disc.grow()
        self.gen.grow()
        self.current_size *= 2
        if self.current_size == 256:
            self.batch_size = 14
        elif self.current_size == 512:
            self.batch_size = 6
        elif self.current_size == 1024:
            self.batch_size = 3

        self.optimizerD = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0, 0.99))
        self.optimizerG = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0, 0.99))

        self.input = torch.FloatTensor(self.batch_size, 3, self.current_size, self.current_size)

        self.cuda()

    def increase_alpha(self, amount):
        self.disc.alpha += amount
        self.gen.alpha += amount

    def set_lr(self, lr):
        for param_group in self.optimizerD.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizerG.param_groups:
            param_group['lr'] = lr

        self.lr = lr

    # Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(self.batch_size, int(real_data.nelement()/self.batch_size)).contiguous().view(self.batch_size, 3, self.current_size, self.current_size)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def train(self, data):
        # Train with real
        self.disc.zero_grad()
        data = data.cuda()
        self.input.resize_as_(data).copy_(data)
        self.label.resize_(self.batch_size).fill_(1)
        inputv = Variable(self.input)
        labelv = Variable(self.label)

        output = self.disc(inputv)
        real_output = output
        errD_real = self.criterion(output, labelv)
        errD_real.backward(retain_graph=True)

        errD_drift = torch.mean((real_output**2)) * 0.001
        errD_drift.backward()

        # Train with fake
        self.noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
        noisev = Variable(self.noise)
        fake = self.gen(noisev)
        self.label.resize_(self.batch_size)
        labelv = Variable(self.label.fill_(0))
        output = self.disc(fake.detach())
        errD_fake = self.criterion(output, labelv)
        errD_fake.backward()

        # Train with gradient penalty
        errD_grad = self.calc_gradient_penalty(self.disc, inputv.data, fake.data)
        errD_grad.backward()

        errD = errD_real + errD_fake + errD_grad + errD_drift
        self.optimizerD.step()

        # Train G
        self.gen.zero_grad()
        labelv = Variable(self.label.fill_(1))
        output = self.disc(fake)
        errG = self.criterion(output, labelv)
        errG.backward()
        self.optimizerG.step()

        return errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], errD_grad.data[0], errD_drift.data[0]


class Generator(nn.Module):
    def __init__(self, start_size=4):
        super(Generator, self).__init__()
        self.blocks = []
        self.size = start_size
        self.alpha = 1.0

        block = []
        block.append(nn.ConvTranspose2d(512, 512, 4))
        block.append(nn.Conv2d(512, 512, 3, padding=1))
        he_init(block[0])
        he_init(block[1])
        self.blocks.append(block)
        self.add_module("block0/0", block[0])
        self.add_module("block0/1", block[1])

        self.toRGB = nn.Conv2d(512, 3, 1)
        he_init(self.toRGB)
        self.add_module("toRGBm", self.toRGB)
        self.toRGB2 = None

        if not start_size == 4:
            for i in range(int(math.log(start_size, 2)-2)):
                self.grow()

    def grow(self):
        self.size = self.size*2
        step = int(math.log(self.size, 2)-1)
        block = []
        if step < 5:
            filters = 512
            block.append(nn.Conv2d(filters, filters, 3, padding=1))
            block.append(nn.Conv2d(filters, filters, 3, padding=1))
        else:
            filters = 512 / (2**(step-4))
            block.append(nn.Conv2d(filters*2, filters, 3, padding=1))
            block.append(nn.Conv2d(filters, filters, 3, padding=1))

        he_init(block[0])
        he_init(block[1])
        self.blocks.append(block)
        self.add_module("block" + str(self.size) + "/0", block[0])
        self.add_module("block0" + str(self.size) + "/1", block[1])

        self.alpha = 1e-6
        self.toRGB2 = self.toRGB
        self.toRGB = nn.Conv2d(filters, 3, 1)
        self.add_module("toRGB" + str(self.size), self.toRGB)
        he_init(self.toRGB)

    def normalize(self, x):
        norm = (x ** 2) + 1e-8
        norm = torch.sum(norm, 1)
        norm = norm / x.size()[1]
        norm = torch.sqrt(norm)

        stack = []
        for i in range(x.size()[1]):
            stack.append(norm)
        norm = torch.stack(stack)
        norm = torch.transpose(norm, 0, 1)

        return x / norm

    def forward(self, x):
        for i in range(len(self.blocks)-1):
            for e in self.blocks[i]:
                x = e(x)
                x = F.leaky_relu(x, negative_slope=0.2)
                if not i == 0:
                    x = self.normalize(x)
            if i == 0:
                x = self.normalize(x)
            if not i == len(self.blocks)-2:
                x = F.upsample(x, scale_factor=2)

        if self.alpha < 1.0:
            x = F.upsample(x, scale_factor=2)
            x1 = self.toRGB2(x) * (1-self.alpha)

            x2 = self.blocks[-1][0](x)
            x2 = F.leaky_relu(x2, negative_slope=0.2)
            x2 = self.normalize(x2)
            x2 = self.blocks[-1][1](x2)
            x2 = F.leaky_relu(x2, negative_slope=0.2)
            x2 = self.normalize(x2)
            x2 = self.toRGB(x2) * self.alpha
            return x1+x2

        else:
            if not len(self.blocks) == 1:
                x = F.upsample(x, scale_factor=2)
            for e in self.blocks[-1]:
                x = e(x)
                x = F.leaky_relu(x, negative_slope=0.2)
                self.normalize(x)
            x = self.toRGB(x)
            return x


class Discriminator(nn.Module):
    def __init__(self, start_size=4):
        super(Discriminator, self).__init__()

        self.blocks = []
        self.size = start_size
        self.alpha = 1.0

        block = []
        block.append(nn.Conv2d(513, 512, 3, padding=1))
        block.append(nn.Conv2d(512, 512, 4))
        he_init(block[0])
        he_init(block[1])
        self.blocks.append(block)
        self.add_module("block0/0", block[0])
        self.add_module("block0/1", block[1])

        self.lin = nn.Linear(512, 1)
        he_init(self.lin, nonlinearity="linear", param=None)
        self.add_module("linear", self.lin)

        self.fromRGB = nn.Conv2d(3, 512, 1)
        self.add_module("fromRGBm", self.fromRGB)
        self.fromRGB2 = None
        he_init(self.fromRGB)

        for i in range(int(math.log(start_size, 2)-2)):
            self.grow()

    def grow(self):
        self.size = self.size*2
        step = int(math.log(self.size, 2)-1)
        block = []
        if step < 5:
            filters = 512
            block.append(nn.Conv2d(filters, filters, 3, padding=1))
            block.append(nn.Conv2d(filters, filters, 3, padding=1))
        else:
            filters = 512 / (2**(step-4))
            block.append(nn.Conv2d(filters, filters*2, 3, padding=1))
            block.append(nn.Conv2d(filters*2, filters*2, 3, padding=1))
        he_init(block[0])
        he_init(block[1])
        self.blocks.append(block)
        self.add_module("block" + str(self.size) + "/0", block[0])
        self.add_module("block" + str(self.size) + "/1", block[1])

        self.alpha = 1e-6
        self.fromRGB2 = self.fromRGB
        self.fromRGB = nn.Conv2d(3, filters, 1)
        self.add_module("fromRGB"+str(self.size), self.fromRGB)
        he_init(self.fromRGB)

    def add_stddev(self, x):
        std = torch.std(x, 0)
        std = torch.mean(std)
        stack = []
        for i in range(x.size()[0]):
            stack.append(std)
        std = torch.stack(stack) # batch_size x 1

        stack = []
        for i in range(x.size()[2]):
            stack.append(std)
        std = torch.stack(stack) # WH x batch_size x 1

        stack = []
        for i in range(x.size()[2]):
            stack.append(std)
        std = torch.stack(stack) # WH x WH x batch_size x 1

        std = torch.transpose(std, 0, 2)
        std = torch.transpose(std, 1, 3)

        return torch.cat([x, std], dim=1)

    def forward(self, x):
        if self.alpha < 1.0:
            x1 = self.fromRGB(x)
            x1 = F.leaky_relu(x1, negative_slope=0.2)
            for e in self.blocks[-1]:
                x1 = e(x1)
                x1 = F.leaky_relu(x1, negative_slope=0.2)
            x1 = F.avg_pool2d(x1, 3, padding=1, stride=2)
            x1 = x1 * self.alpha

            x2 = F.avg_pool2d(x, 3, padding=1, stride=2)
            x2 = self.fromRGB2(x2)
            x2 = F.leaky_relu(x2, negative_slope=0.2)
            x2 = x2 * (1-self.alpha)

            x = x1 + x2

        else:
            x = self.fromRGB(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            if not len(self.blocks) == 1:
                for e in self.blocks[-1]:
                    x = e(x)
                    x = F.leaky_relu(x, negative_slope=0.2)
                x = F.avg_pool2d(x, 3, padding=1, stride=2)

        for i in range(1, len(self.blocks)-1):
            rev = self.blocks[::-1]
            for e in rev[i]:
                x = e(x)
                x = F.leaky_relu(x, negative_slope=0.2)
            x = F.avg_pool2d(x, 3, padding=1, stride=2)
        x = self.add_stddev(x)
        for e in self.blocks[0]:
            x = e(x)
            x = F.leaky_relu(x, negative_slope=0.2)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x = self.lin(x)
        x = torch.squeeze(x)
        return x


# Taken from https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
def he_init(layer, nonlinearity='leaky_relu', param=0.2):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)
