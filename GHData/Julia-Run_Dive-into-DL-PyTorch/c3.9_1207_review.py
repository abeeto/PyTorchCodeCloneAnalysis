# 多重感知机理的简洁实现
import torch
import torch.nn as nn
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

# data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# in out
num_in = 28 * 28
num_h = 256
num_out = 10

# net
net = nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_in, num_h), nn.ReLU(), nn.Linear(num_h, num_out))

# init
for param in net.parameters():
    nn.init.normal_(param, 0, 0.01)

# LOSS
loss = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 3

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
