import torch
import numpy as np
from d2lzh_pytorch.utils import *

num_epochs = 100

# 生成训练测试数据和标签
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))  # 特征x
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)  # 特征 x x的平方 x的三次方
#  torch.cat()函数可以将多个张量拼接成一个张量。torch.cat()有两个参数，第一个是要拼接的张量的列表或是元组；第二个参数是拼接的维度。
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加入噪声

print(features[:2], poly_features[:2], labels[:2])  # 生成的特征和标签的前两个样本。

# 定义损失函数
loss = torch.nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))  # 计算损失
            optimizer.zero_grad()            # 梯度清零
            l.backward()                     # 计算梯度
            optimizer.step()                 # 优化参数
        train_labels = train_labels.view(-1, 1)  # 转换为长度行一列
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


# 三阶多项式函数拟合（正常）  y=1.2x−3.4x**2+5.6x**3+5
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
# 线性函数拟合（欠拟合） x全是1次方
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
# 训练样本不足（过拟合） 只使用两个样本来训练模型
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])
