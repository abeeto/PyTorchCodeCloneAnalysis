import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

# training data set and other parameters
true_w = [2, -3.4]
true_b = 4.2
num_examples = 1000
num_inputs = 2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, labels.size()))


# print(features[0], labels[0])

def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # unit inches
    # 显示并设置图片大小
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize  # 类似disk


set_figsize((8, 6))
plt.scatter(features[:, 1].numpy(), labels.numpy(), 10)


# plt.show()


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):  # ordered,indices的index
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 切片，无序
        yield features.index_select(0, j), labels.index_select(0, j)  # 10组数据，(X,y)


#
# batch_size = 10
#
# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

# initial data w and b 给定参数的初始值
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
print(w, b, sep='-----')


# define  model
def linreg(X, w, b):
    return torch.mm(X, w) + b  # 广播效应，输出值为y_hat,


# define loss func
def squared_loss(y_hat, y):  # tensor, labels
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# define optimize algorithm
def sgd(params, lr, batch_size):
    # params包含w，b
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)  # 全部data set的loss：1000X1的tensor
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
print(true_w, '\n', w)
print(true_b, '\n', b)
