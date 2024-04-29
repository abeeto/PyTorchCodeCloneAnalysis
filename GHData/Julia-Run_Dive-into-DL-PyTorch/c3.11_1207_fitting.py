# 模型拟合，欠拟合，过拟合
# note
# 改变参数，训练数据集上更准确，测试数据集上不一定更准确
# training error, generalization error (训练误差，泛化误差)
# 前者为在训练数据集上的误差，后者为在测试数据集上的误差期望。（高三学生做高考题，不一定会做的很好）
################# 机器学习模型应关注降低泛化误差。
# 本书中，test set 为 validation set，， 测试结果也为验证结果。
# --k-fold cross-validation: k-1,training,1 test -- k times --error(train/test).mean()

# underfitting, overfitting--(train_error too large / training error << test_error )
# reasons: 模型复杂度，training set 大小，。。。。。。
# 以自己生成的数据集和多项式函数做实验

# y=1.2x−3.4x^2 +5.6x^3 +5+ϵ,

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l

# y=1.2x−3.4x^2 +5.6x^3 +5+ϵ,
true_w = [1.2, -3.4, 5.6]
true_b = 5
n_train = 100
n_test = 100
f = torch.randn((n_train + n_test, 1), dtype=torch.float32)
poly_features = torch.cat((f, torch.pow(f, 2), torch.pow(f, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
print(f[:2], '\n', poly_features[:2], '\n', labels[:2])


def semilogy(x, y, xl, yl, x2=None, y2=None, legend=None, figsize=(7, 5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(xl)
    d2l.plt.ylabel(yl)
    d2l.plt.semilogy(x, y)
    if x2 and y2:
        d2l.plt.semilogy(x2, y2, linestyle=':')
        d2l.plt.legend(legend)


num_epochs, loss = 100, torch.nn.MSELoss()


def fit_and_plot(train_f, test_f, train_l, test_l):
    net = torch.nn.Linear(train_f.shape[-1], 1)  # train_f:batch_s,num_in
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    batch_size = min(10, train_l.shape[0])  # train_l: batch_size,num_out
    dataset = torch.utils.data.TensorDataset(train_f, train_l)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_l = train_l.view(-1, 1)
        test_l = test_l.view(-1, 1)
        train_ls.append(loss(net(train_f), train_l).item())
        test_ls.append(loss(net(test_f), test_l).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])
plt.show()
fit_and_plot(f[:n_train, :], f[n_train:, :], labels[:n_train],
             labels[n_train:])
plt.show()
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])
plt.show()
