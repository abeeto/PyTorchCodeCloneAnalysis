import torch
import torchvision
import numpy as np
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

# DATA/iter
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# init w b
num_inputs = 28 * 28  ##w:(28*28)*10, b:1*10
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# softmax----yhat exp,partition
def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


# module  return yhat here softmax are used
def net(x):  # ------yhat
    return softmax(torch.mm(x.view(-1, num_inputs), w) + b)


# loss cross_entropy. only the label = 1, * (-log(yhat))
def cross_entropy(yhat, y):
    return -torch.log(yhat.gather(1, y.view(-1, 1)))  #
    # gather：yview当前值的位置为1，其他为0


# for test
ytt = torch.LongTensor([0, 2])
print(ytt.view(-1, 1))


# accuracy
def accuracy(yhat, y):
    return (yhat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0  # n代表总iter次数，即data的总batch
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))

epochs = 5
lr = 0.1
batch_size = 10


def train_ch3(net, train_iter, test_iter, loss, ecochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            yhat = net(x)
            l = loss(yhat, y).sum()  # loss

            if optimizer is not None:  # 梯度清零，针对不同情况
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()  # back

            if optimizer is None:  # 更新参数
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()  # 累计loss误差
            train_acc_sum += (yhat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, epochs, batch_size, [w, b], lr)

# trainning结束，得到10个线性方程，各自有28*28个输入，该方程组为学习的模型
# test时，yhat.argmax函数可以定位到概率最大的方程输出，对应labels

# # test
x, y = iter(test_iter).next()
true_label = d2l.get_fashion_mnist_labels(y.numpy())
label_hat = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_label, label_hat)]
d2l.show_fashion_mnist(x[0:9], titles[0:9])
