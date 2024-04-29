from collections import OrderedDict
import d2lzh as d2l
import torch
from torch import nn
from torch.nn import init
# import sys

# sys.path.append("/home/kesci/input")

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='./dataset')

num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

# net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    # FlattenLayer(),
    # LinearNet(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
        ('linear', nn.Linear(num_inputs, num_outputs))])
)
# init parms
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              batch_size, None, None, optimizer)

# prediction
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
