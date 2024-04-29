from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from operator import itemgetter

from dataset import PalindromeDataset
from lstm import LSTM

def train(config):

    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)  # fixme

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        hit = 0;
        n, dim = batch_inputs.size()
        batch_inputs_T = torch.transpose(batch_inputs, 0, 1)

        y_hat_oh = model.forward(batch_inputs_T)
        for i in range(n):
            y_pre, _ = max(enumerate(y_hat_oh[i]), key=itemgetter(1))
            y = batch_targets[i].item()
            if y_pre == y:
                hit += 1

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        # Add more code here ...

        loss = criterion(y_hat_oh, batch_targets)  # fixme
        accuracy = hit / n * 100  # fixme

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print("loss: ", loss.item())
            print("accuracy: ", accuracy)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)