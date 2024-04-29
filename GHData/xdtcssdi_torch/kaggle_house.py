import torch
from torch import nn
import pandas as pd


def createDatasets():
    """
    创建数据集
    :return:
    """
    train_data = pd.read_csv("kaggle_house/train.csv")  # 获取训练数据
    test_data = pd.read_csv("kaggle_house/test.csv")  # 获取测试数据
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32).view(-1, 1)

    train_n = train_data.shape[0]
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), axis=0)

    # 将数据标准化
    idx = all_features.dtypes[all_features.dtypes != "object"].index  # 获取数据化数据的表头

    all_features[idx] = all_features[idx].apply(lambda x: (x - x.mean()) / x.std())  # 标准化
    all_features[idx] = all_features[idx].fillna(0)  # 填充缺省值

    test_data[idx[:-1]] = test_data[idx[:-1]].apply(lambda x: (x - x.mean()) / x.std())  # 标准化
    test_data[idx[:-1]] = test_data[idx[:-1]].fillna(0)  # 填充缺省值

    all_features = pd.get_dummies(all_features, dummy_na=True)

    return torch.tensor(all_features.iloc[:train_n, :].values, dtype=torch.float32), train_labels, torch.tensor(
        all_features.iloc[train_n:, :].values, dtype=torch.float32)


# 设置均方误差
loss = nn.MSELoss()


def get_net(feature_num):
    """
    创建单层网络并且初始化参数
    :param feature_num:
    :return:
    """
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    """
    对数均方根误差
    :param net: 
    :param features: 
    :param labels: 
    :return: 
    """
    with torch.no_grad():
        clipped_labels = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_labels.log(), labels.log()))  # 利用均方误差实现对数均方根误差
        return rmse.item()


def train(net, train_features, train_labels, valid_features, valid_labels, num_epoch, lr, weight_decay, batch_size):
    """
    由于Ｋ折交叉验证，所以每次都得创建数据集和优化器
    :param net:
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :param num_epoch:
    :param lr:
    :param weight_decay:
    :param batch_size:
    :return:
    """
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_ls, valid_ls = [], []
    for epoch in range(1, num_epoch + 1):
        for X, y in train_iter:
            l = loss(net(X), y)
            net.zero_grad()
            l.backward()
            opt.step()
        # 计算总的训练误差
        train_l = log_rmse(net, train_features, train_labels)
        train_ls.append(train_l)
        # 如果有验证集,计算在验证集上的损失误差
        if valid_features is not None:
            valid_l = log_rmse(net, valid_features, valid_labels)
            valid_ls.append(valid_l)
    return train_ls, valid_ls


def get_k_fold_data(k, i, X, y):
    """
    ｋ折分割数据集
    １份作为验证集，其余组合为训练集
    :param k:
    :param i:
    :param X:
    :param y:
    :return:
    """
    assert i <= k
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = [None, ] * 4
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epoch, lr, weight_decay, batch_size):
    train_sum, valid_sum = 0, 0
    for i in range(k):
        # 对每一折训练
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epoch, lr, weight_decay, batch_size)
        train_sum += train_ls[-1]
        valid_sum += valid_ls[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))


if __name__ == '__main__':
    train_data, train_labels, test_data = createDatasets()
    k_fold(5, train_data, train_labels, num_epoch=100, lr=5.0, weight_decay=0, batch_size=64)
