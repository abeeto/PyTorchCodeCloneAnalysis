
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.datasets import mnist

import os, sys, random

from modules import TrainableModel

from scipy.misc import imresize
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import IPython


class Network(TrainableModel):

    def __init__(self):
        super(Network, self).__init__()
        self.inputs = ["tophalf", "zoomed"]
        self.targets = ["digit", "parity", "curvature", "floating"]

        self.hidden = nn.Linear(280, 128)
        
        self.encode1 = nn.Linear(80, 128)
        self.out1 = nn.Linear(128, 10)
        self.out2 = nn.Linear(128, 2)
        self.out3 = nn.Linear(128, 3)
        self.out4 = nn.Linear(128, 1)

    def loss(self, data, data_pred):

        loss = 0.0
        loss += F.nll_loss(data_pred['digit'], data['digit'])
        loss += F.nll_loss(data_pred['parity'], data['parity'])
        loss += F.nll_loss(data_pred['curvature'], data['curvature'])
        loss += nn.MSELoss()(data_pred['floating'][:, 0], data['floating'])
        return loss

    def score(self, preds, targets):

        scores = {}
        for target in ["digit", "parity", "curvature"]:
            score = accuracy_score(np.argmax(preds[target], axis=1), targets[target])
            base_score = accuracy_score(np.argmax(preds[target], axis=1), shuffle(targets[target]))
            scores[target] = "{0:.4f}/{1:.4f}".format(score, base_score)

        score, pval = pearsonr(preds["floating"][:, 0], targets["floating"])
        scores["floating"] = "{0:.3f}(p={1:.4E})".format(score, pval)

        return scores

    def forward(self, data):

        if self.training:
            if random.choice([True, False]):
                del data[random.choice(data.keys())]

        x, y = None, None
        if "tophalf" in data:
            x = data["tophalf"]
            x = x.view(x.size(0), -1)
            x = F.relu(self.hidden(x))

        if "zoomed" in data:
            y = data["zoomed"]
            y = y.unsqueeze(1)
            y = F.relu(self.conv(y))
            y = F.dropout(F.max_pool2d(y, 2), p=0.3, training=self.training)
            y = F.relu(self.conv2(y))
            y = y.view(y.size(0), -1)
            y = F.relu(self.encode1(y))

        if x is not None and y is not None:
            x = x + y
        elif y is not None:
            x = y

        x = F.dropout(x, p=0.4, training=self.training)
        
        out1 = F.log_softmax(self.out1(x))
        out2 = F.log_softmax(self.out2(x))
        out3 = F.log_softmax(self.out3(x))
        out4 = F.sigmoid(self.out4(x))

        return {"digit": out1, "parity": out2, "curvature": out3, "floating": out4}


"""Train model on MNIST dataset."""

model = Network()
model.compile(optim.Adadelta, lr=0.1)
print (model)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train.astype(float), X_test.astype(float)
Y_train, Y_test = Y_train.astype(int), Y_test.astype(int)

def datagen(X, Y):
    for i, (data, target) in enumerate(zip(X, Y)):
        if i % 1000 == 0: print (i)
        tophalf = data[0:28, 0:10]
        zoomed = imresize(data, (8, 8))

        target2 = target % 2
        target3 = 0 if target in [1, 7, 4] else 1 if target in [5, 7, 2] else 2
        target4 = target/10.0**(0.8)

        if random.randint(0, 1) == 0:
            yield {"tophalf": tophalf, "zoomed": zoomed, "digit": target, "parity": target2, "curvature": target3}
        else:
            yield {"tophalf": tophalf, "zoomed": zoomed, "digit": target, "floating": target4}

for epochs in range(0, 40000):
    
    train_gen = datagen(X_train, Y_train)
    test_gen = datagen(X_test, Y_test)
    model.fit(train_gen, test_gen, batch_size=128)

