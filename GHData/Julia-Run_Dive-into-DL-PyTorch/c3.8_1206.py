import torch
import numpy as np
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_hiddens = 256
num_outputs = 10

w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float32)
b1 = torch.zeros(num_hiddens, dtype=torch.float32)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float32)
b2 = torch.zeros(num_outputs, dtype=torch.float32)

params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# actiive
def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))


# model
def net(x):
    x = x.view(-1, num_inputs)  # 256,in
    h = relu(torch.matmul(x, w1) + b1)  # 256,hid
    o = torch.matmul(h, w2) + b2  # 256,out
    return o


# loss
loss = torch.nn.CrossEntropyLoss()

num_epo = 5
lr = 100.0

d2l.train_ch3(net, train_iter, test_iter, loss, num_epo, batch_size, params, lr)
