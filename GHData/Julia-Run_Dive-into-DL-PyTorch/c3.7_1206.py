import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

# data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

for x, y in train_iter:
    print(x.shape)  # 256,1,28,28
    print(y.shape)  # 256
    break
# model
num_inputs = 28 * 28
num_outputs = 10


class LinearNet(nn.Module):  # M大写
    def __init__(self, num_in, num_out):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_in, num_out)

    def forward(self, x):
        return self.Linear(x.shape[0], -1)  # 输入的形式
        # x--256,28*28   y---256,10. w--10,28*28   b--10,1


a = torch.rand(10, 2, 3, 4)
print(a.shape[0])

net = LinearNet(num_inputs, num_outputs)


class FlatternLayer(nn.Module):
    def __init__(self):
        super(FlatternLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


from collections import OrderedDict

net = nn.Sequential(OrderedDict([('flatten', FlatternLayer()), ('linear', nn.Linear(num_inputs, num_outputs))]))

# init
w = nn.init.normal_(net.linear.weight, 0, 0.01)  # linear 10,28*28
b = nn.init.zeros_(net.linear.bias) # 10,1

loss = nn.CrossEntropyLoss()  # 对与softmax而言的loss--CrossEntropyLoss  # batch_size * 1

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 需要输入参数

epoch = 3
# d2l.train_ch3(net, train_iter, test_iter, loss, epoch, batch_size, None, None, optimizer)  # None--  params，lr
# 定义了optimizer，所以不需要params，lr。这两个在optimizer里有
d2l.train_ch3(net, train_iter, test_iter, loss, epoch, batch_size, [w, b], 0.1)
x, y = iter(test_iter).next()  # 用法
true_labels = d2l.get_fashion_mnist_labels(y)
prediction = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1))
titles = [true + '\n' + pre for true, pre in zip(true_labels, prediction)]
d2l.show_fashion_mnist(x[0:9], titles[0:9])
