import logging
import time
import numpy as np
import torch as torch
from torch import nn
import torch.optim as optim
import tensorflow as tf


DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.model = NonsharedModel(config, bsde)
        self.y_init = self.model.y_init
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.net_config.lr_values, eps= 1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=self.net_config.gamma)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            self.optimizer.zero_grad()
            train_data = self.bsde.sample(self.net_config.valid_size)
            loss = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data)
                y_init = self.y_init.detach().numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
        return np.array(training_history)

    def loss_fn(self, inputs):
        dw, x = inputs
        y_terminal = self.model(inputs)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, torch.square(delta),
                                       2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))

        return loss


class NonsharedModel(nn.Module):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = nn.Parameter(torch.DoubleTensor(1).uniform_(self.net_config.y_init_range[0],
                                    self.net_config.y_init_range[1])
                                    , requires_grad=True)
        self.z_init = nn.Parameter(torch.DoubleTensor(1, self.eqn_config.dim).uniform_(-.1,
                                        .1)
                                    , requires_grad=True)

        self.subnet = FeedForwardSubNet(config)
        self.subnet.apply(self.weights_init)
        
         # custom weights initialization
    def weights_init(self,m):
            classname = m.__class__.__name__
            if classname.find('bn') != -1:
                m.weight.data.uniform_(0.1, 0.5) #gamma
                m.bias.data.normal_(0,0.1) #beta

    def forward(self, inputs):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = torch.ones([ list(dw.size())[0], 1], dtype=torch.float64)
        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, self.z_init)
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)
            z = self.subnet(x[:, :, t + 1])
            z = z/self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            torch.sum(z * dw[:, :, -1], 1, keepdims=True)
        return y


class FeedForwardSubNet(nn.Module):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        model = nn.Sequential()
        layer_struct = config.net_config.layer_struct
        layers = []
        model.add_module("bn_initial",nn.BatchNorm1d(layer_struct[0], momentum=0.99, eps=1e-6))
        for i in range(len(layer_struct)-1):
            model.add_module("linear{0}".format(i), nn.Linear(layer_struct[i], layer_struct[i+1], bias=False))
            model.add_module("bn{0}".format(i), nn.BatchNorm1d(layer_struct[i+1], momentum=0.99, eps=1e-6))
            model.add_module("relu{0}".format(i), nn.ReLU())
        model.add_module("bn_last", nn.BatchNorm1d(layer_struct[-1], momentum=0.99, eps=1e-6))
        self.model = model

    def forward(self, x):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        output = self.model(x)
        return output
