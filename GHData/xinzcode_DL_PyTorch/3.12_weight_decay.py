import torch
import numpy as np
from d2lzh_pytorch.utils import *

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = linreg, squared_loss  # 定义网络和损失函数

features = torch.randn((n_train + n_test, num_inputs))  # 特征
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 标签
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


# # 初始化模型参数
# def init_params():
#     w = torch.randn((num_inputs, 1), requires_grad=True)
#     b = torch.zeros(1, requires_grad=True)
#     return [w, b]
#
#
# # L2范数惩罚项
# def l2_penalty(w):
#     return (w**2).sum() / 2
#
#
# # 训练模型
# def fit_and_plot(lambd):
#     w, b = init_params()
#     train_ls, test_ls = [], []
#     for _ in range(num_epochs):
#         for X, y in train_iter:
#             # 添加了L2范数惩罚项
#             l = loss(net(X, w, b), y) + lambd * l2_penalty(w)  # lambd为超参数λ。当λ设为0时，惩罚项完全不起作用。
#             l = l.sum()
#
#             if w.grad is not None:
#                 w.grad.data.zero_()
#                 b.grad.data.zero_()
#             l.backward()
#             sgd([w, b], lr, batch_size)
#         train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
#         test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
#     semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
#                  range(1, num_epochs + 1), test_ls, ['train', 'test'])
#     print('L2 norm of w:', w.norm().item())
#
#
# fit_and_plot(lambd=0)
#
# fit_and_plot(lambd=3)


# 简洁实现
def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)  # 通过weight_decay参数来指定权重衰减超参数。
    # 默认下，PyTorch会对权重和偏差同时衰减。我们可以分别对权重和偏差构造优化器实例，从而只对权重衰减。
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


fit_and_plot_pytorch(0)

fit_and_plot_pytorch(3)
