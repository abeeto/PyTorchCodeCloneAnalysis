import torch
import numpy as np
import torch.nn as nn
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_hidden = 256
num_outputs = 10

net = nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_inputs, num_hidden), nn.ReLU(),
                    nn.Linear(num_hidden, num_outputs))
# x--in shape(batch_size,num_input), + linear + active + linear ......(layer by layer, ans sequential)
for params in net.parameters():
    nn.init.normal_(params, 0, 0.01)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
epo = 5
d2l.train_ch3(net, train_iter, test_iter, loss, epo, batch_size, None, None, optimizer)
