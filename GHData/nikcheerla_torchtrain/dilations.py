import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from keras.datasets import mnist, cifar10

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
        self.inputs = ["data"]
        self.targets = ["target"]

        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 16, (5, 5), padding=1)
        self.conv3 = nn.Conv2d(16, 8, (5, 5), padding=1)

        self.hidden = nn.Linear(200, 128)
        self.output = nn.Linear(128, 10)

    def loss(self, data, data_pred):
        
        loss = F.nll_loss(data_pred['target'], data['target'])
        return loss

    def auxilary_loss(self):

    	checkerboard_loss = 0.0
    	for conv_layer in [self.conv1, self.conv2, self.conv3]:
    		checkerboard_loss += conv_layer.weight[:, :, 0:5:2, 0:5:2].mean()
    	return checkerboard_loss

    def score(self, preds, targets):

        score = accuracy_score(np.argmax(preds["target"], axis=1), targets["target"])
        base_score = accuracy_score(np.argmax(preds["target"], axis=1), shuffle(targets["target"]))
        return "{0:.4f}/{1:.4f}".format(score, base_score)

    def forward(self, data):

    	x = data["data"]
    	
    	x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 4)

        """
    	x = F.relu(self.conv_list[0](x))
    	x = F.dropout(x, p=0.3, training=self.training)
        """
        """
    	for conv_layer in self.conv_list:
    		x = F.relu(conv_layer(x))
        """

    	#x = F.max_pool2d(x, 2)
    	x = x.view(x.size(0), -1)
    	x = F.sigmoid(self.hidden(x))
    	x = F.dropout(x, p=0.5, training=self.training)
    	x = F.softmax(self.output(x))

        return {"target": x}





"""Train model on MNIST dataset."""

model = Network()
model.compile(optim.Adadelta, lr=0.1)
print (model)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train.astype(float), X_test.astype(float)
Y_train, Y_test = Y_train.astype(int), Y_test.astype(int)

def datagen(X, Y):
    for i, (data, target) in enumerate(zip(X, Y)):
    	#if i % 1000 == 0: print (i)
        yield {"data": data, "target": target}

for epochs in range(0, 40000):
    
    train_gen = datagen(X_train, Y_train)
    test_gen = datagen(X_test, Y_test)
    model.fit(train_gen, test_gen, batch_size=128)
