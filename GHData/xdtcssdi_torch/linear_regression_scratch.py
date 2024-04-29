import random
import torch
import numpy as np

def createDatas(number_sample, features):
    """
    create dateset by number_sample, features
    :return: train_data, label
    """
    w = [3.5, 1.3]
    b = 1.5
    X = torch.randn(number_sample, features, dtype=torch.float32)

    label = X[:, 0] * w[0] + X[:, 1] * w[1] + b

    label += torch.tensor(np.random.normal(0, 0.01, size=label.shape), dtype=torch.float32)
    return X, label

def date_iters(datasets, batch_size):
    """
    返回batchsize的数据集
    :param datasets:
    :param batch_size:
    :return: yield
    """
    X, y = datasets
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        index = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield X.index_select(0, index), y.index_select(0, index)


def net(X, w, b):
    """
    线性回归模型
    :param X:
    :param w:
    :param b:
    :return:
    """
    return torch.mm(X, w) + b

def square_loss(y_true, y_predict):
    """
    平方误差损失
    :param y_true:
    :param y_predict:
    :return:
    """

    return (y_true - y_predict.view(y_true.shape)) ** 2 / 2


def update_parames(params, lr, m):
    """
    更新优化参数
    :param params:
    :param lr:
    :param m:
    :return:
    """
    for param in params:
        param.data -= lr * param.grad.data/m


if __name__ == '__main__':
    num_inputs = 2
    num_examples = 1000
    batch_size = 10
    lr = 0.03
    num_iters = 3
    w = torch.tensor(np.random.normal(0, 0.01, size=[2, 1]), requires_grad=True, dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

    X, y = createDatas(num_examples, num_inputs)
    data_iter = date_iters((X, y), batch_size)
    for i in range(num_iters):
        for x, y_ in data_iter:
            y_predict = net(x, w, b)
            loss = square_loss(y_, y_predict)
            loss.backward(torch.ones_like(loss))

            update_parames([w, b], lr, batch_size)

            w.grad.data.zero_()
            b.grad.data.zero_()

    print(w, b)


