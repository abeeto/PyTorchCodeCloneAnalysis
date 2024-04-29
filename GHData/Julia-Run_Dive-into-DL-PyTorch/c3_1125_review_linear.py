import torch
from matplotlib import pyplot as plt
import random
import numpy as np
from IPython import display

# 创建training数据集
# 正确的w，b，fretures，labels（with noise），样本总量
true_w = [4.3, 2.5]
true_b = 3.1
num_examples = 1000
num_inputs = 2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.1, labels.size()))

print(len(features))


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  # 0-999
    random.shuffle(indices)  # indices排序打乱
    for i in range(0, num_examples, batch_size):  # 总数分批
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        # 对乱序的indices切片，切片长batch—size ---内容：0-999之间的乱序数字
        yield features.index_select(0, j), labels.index_select(0, j)


# 初始化
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
# 正太分布
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
print(w, b, sep='-----')


# 定义model：
def linreg(X, w, b):
    return torch.mm(X, w) + b


# loss function
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# optomization function
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# experiment
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss_now = loss(net(X, w, b), y).sum()  # 一定要有sum
        loss_now.backward()  # 得到param.grad的值
        sgd([w, b], lr, batch_size)  # 更新param的数据

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean()))
print(true_w, '\n', w)
print(true_b, '\n', b)
