from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import sklearn.datasets
import sklearn.preprocessing
import torch
from operator import itemgetter

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 100

FLAGS = None

def accuracy(predictions, targets, mlp):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    result = mlp.forward(predictions)
    num = list(result.size())
    hit = 0
    for i in range(num[0]):
        crt = list(result[i])
        aim = targets[i].item()
        index, _ = max(enumerate(crt), key=itemgetter(1))
        if index == aim:
            hit += 1
    accuracy = hit / num[0] * 100
    return accuracy

def train(mlp, x, y, args):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    y_pred = mlp(x)
    loss = mlp.criterion(y_pred, y)
    # print(loss.item())
    mlp.optimizer.zero_grad()
    loss.backward()
    mlp.optimizer.step()
    return mlp

def main(args):
    """
    Main function
    """
    x, y = sklearn.datasets.make_moons(n_samples=1000)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    predictions, targets = sklearn.datasets.make_moons(n_samples=200)
    predictions = torch.from_numpy(predictions).float()
    targets = torch.from_numpy(targets).long()

    hiddens_temp = args.dnn_hidden_units.split(",")
    hiddens = list()
    for value in hiddens_temp:
        hiddens.append(int(value))
    mlp = MLP(2, hiddens, 2)
    mlp.optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-4)
    # print("create finish")

    for i in range(args.max_steps):
        mlp = train(mlp, x, y, args)
        if i % args.eval_freq == 0:
            print(accuracy(predictions, targets, mlp))
    return

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args)