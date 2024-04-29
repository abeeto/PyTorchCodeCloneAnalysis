# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import numpy as np
from Dataset import *

d = Dataset(784, 10)

model = torch.load("model")

i = 0
while (i <= 9):
    y_pred = model(Variable(torch.Tensor(d.imageToArray("minimnist/" + str(i) + ".png", "L"))))
    # print y_pred
    print str(i) + " Index max: " + str(np.argmax(y_pred.data.numpy()))
    i += 1

# print model

