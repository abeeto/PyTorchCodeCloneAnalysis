import torch
import torchvision
import numpy as np
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

# data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
for x, y in train_iter:
    print(type(x), x.shape)
    print(type(y), y.shape)
    break

# init
num_inputs = 28 * 28
num_outputs = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32)
b = torch.zeros(10, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# softmax


def softmax(x):
    x_exp = x.exp()  # ===x_exp=torch.exp(x),,,,where x is a tensor
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


# x = torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.6, 0.2]])
# print(x.sum(dim=1))
# print(x.sum(dim=1,keepdim=True))
# model
def net(x):  # -----yhat
    return softmax(torch.mm(x.view(-1, num_inputs), w) + b)


# loss
# update
# cross_entropy
def cross_entropy(yhat, y):
    return -torch.log(yhat.gather(1, y.view(-1, 1)))  # 和有view（-1）有区别（2d--1d）
    # # y.view(-1, 1)对应着每组输入的正确输出的index，=1，其他为0


# accuracy
def accuracy(yhat, y):
    return (yhat.argmax(dim=1) == y).float().mean().item()
    # yhat dim=1 上的最大值的index，构成的矩阵是否和y相等


def evaluate_accuracy(data_iter, net):  # 对整体的预估
    error_acc, k = 0.0, 0
    for x, y in data_iter:
        true = y
        yhat = net(x)
        error_acc += (yhat.argmax(dim=1) == true).float().sum().item()
        k += y.shape[0]  ############################
    return error_acc / k


epochs = 5
lr = 0.1
batch_size = 256


def train_ch3(net, train_iter, test_iter, loss, epochs, batch_size, params=None, lr=None, optimizer=None):
    # 在每轮的循环中：计算yhat，计算损失（w，b），梯度data清零，back，更新参数
    for i in range(epochs):
        train_loss_sum, train_acura_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            yhat = net(x)
            l = loss(yhat, y).sum()  #################################

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is not None:
                optimizer.step()
            else:
                d2l.sgd(params, lr, batch_size)

            train_loss_sum += l.item()
            train_acura_sum += (yhat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]  #################################################
        test_acc_sum = evaluate_accuracy(test_iter, net)
        print('epoch = %d, loss = %.5f, train accuracy = %.5f, test accuracy = %.5f' % (
            i + 1, train_loss_sum / n, train_acura_sum / n, test_acc_sum))


train_ch3(net, train_iter, test_iter, cross_entropy, epochs, batch_size, [w, b], lr)
# x, y = iter(test_iter).next()
# true_label = d2l.get_fashion_mnist_labels(y.numpy())
# label_hat = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
# titles = [true + '\n' + pred for true, pred in zip(true_label, label_hat)]
# d2l.show_fashion_mnist(x[0:9], titles[0:9])
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

import matplotlib.pyplot as plt
def show_fashion_mnist(images, labels):  # images---list（features）， labels--list
    # 下面定义一个可以在一行里画出多张图像和对应标签的函数。
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 1Xlen()个子图
    for f, img, lbl in zip(figs, images, labels):  # zip(*zip()) 可迭代元素打包
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)  # label
        f.axes.get_xaxis().set_visible(False)  # 不显示坐标轴的刻度
        f.axes.get_yaxis().set_visible(False)
    plt.show()


show_fashion_mnist(X[0:9], titles[0:9])  # d2l.show......似乎没有show（）
