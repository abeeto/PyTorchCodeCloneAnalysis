"""
Author: Qizhi Li
RNN模型
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import numpy as np
# from preprocess import preprocess  # linux上使用
from rnn_based_sa_ch import preprocess


class RNN(nn.Module):

    def __init__(self, num_hiddens, num_inputs, num_outputs, bidirectional=False,
                 num_layers=1):
        super().__init__()
        self.num_hiddens_rnn = num_hiddens
        self.num_hiddens_linear = (2 * num_hiddens) if bidirectional else num_hiddens
        self.rnn_layer = nn.RNN(input_size=num_inputs, hidden_size=self.num_hiddens_rnn,
                                batch_first=True, bidirectional=bidirectional,
                                num_layers=num_layers)
        self.linear = nn.Linear(in_features=self.num_hiddens_linear, out_features=num_outputs)
        self.softmax = nn.Softmax()
        self.state = None

    def forward(self, input, state):
        """
        前向传播
        :param input: tensor
                shape: (batch_size, max_seq_length, w2v_dim)
                输入数据
        :param state: tensor
                shape: (num_layers, batch_size, num_outputs)
                隐藏层状态
        :return output: tensor
                shape: (batch_size, num_outputs)
                输出结果
        :return state: tensor
                shape: (num_layers, batch_size, num_outputs)
                隐藏层状态
        """
        # rnn_y shape: (batch_size, seq_length, w2v_dim)
        rnn_y, self.state = self.rnn_layer(input, state)
        rnn_last_y = rnn_y[:, -1, :]
        linear_y = self.linear(rnn_last_y.view(-1, rnn_last_y.shape[-1]))
        output = self.softmax(linear_y)

        return output, self.state


def get_k_fold_data(k, i, X, y):
    """
    获得第i折交叉验证所需要的数据
    :param k: int
            交叉验证的折数
    :param i: int
            第i轮交叉验证
    :param X: tensor
            shape: (num_seq, seq_length, w2v_dim)
            输入数据
    :param y: tensor
            shape: (num_seq, )
    :return X_train: tensor
            shape: ((num_seq // k) * (k - 1), seq_length, w2v_dim)
            第i折训练数据
    :return y_train: tensor
            shape: ((num_seq // k) * (k - 1), )
            第i折训练标签
    :return X_valid: tensor
            shape: (num_seq // k, seq_length, w2v_dim)
            第i折验证数据
    :return y_valid: tensor
            shape: (num_seq // k, seq_length, w2v_dim)
            第i折验证标签
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 获得元素切片
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            # 如果是第i折数据, 则这部分数据为验证集的数据
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            # 如果不是第i折数据, 且训练集为空, 则这部分数据为训练集第一部分数据
            X_train, y_train = X_part, y_part
        else:
            # 如果不是第i折数据, 且训练集不为空, 则这部分数据拼接到训练集中
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid


def get_accuracy(y_hat, y):
    """
    判断预测准确率
    :param y_hat: tensor
            shape: (batch_size, num_outputs)
            预测数据
    :param y: tensor
            shape: (batch_size, )
            真实数据
    :return: float
            准确率
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def train(model, batch_size, X_train, y_train, X_test, y_test, lr, num_epochs,
          weight_decay):
    """
    训练数据
    :param model: Object
            模型的实例化对象
    :param batch_size: int
            每个batch的大小
    :param X_train: tensor
            shape: ((num_seq // k) * (k - 1), seq_length, w2v_dim)
            训练数据
    :param y_train: tensor
            shape: ((num_seq // k) * (k - 1), )
            训练标签
    :param X_test: tensor
            shape: (num_seq // k, seq_length, w2v_dim)
            测试数据
    :param y_test: tensor
            shape: (num_seq // k, )
            测试标签
    :param lr: float
            学习率
    :param num_epochs: int
            迭代次数
    :param weight_decay: float
            权重衰减参数
    :return : float
            该折训练集的平均损失
    :return : float
            该折训练集的平均准确率
    :return : float
            该折测试集的平均损失
    :return : float
            该折测试集的平均准确率
    """
    state = None
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)
    train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 drop_last=True)
    test_iter = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=True)

    train_ls_sum, train_acc_sum, test_ls_sum, test_acc_sum = [], [], [], []
    for epoch in range(num_epochs):
        start = time.time()
        train_ls, test_ls, train_acc, test_acc = 0.0, 0.0, 0.0, 0.0
        train_n = 0

        model.train()
        for X, y in train_iter:
            if state is not None:
                state = state.detach()

            # y_hat: shape (batch_size, num_outputs)
            y_hat, state = model(X, state)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_ls += l.item()
            train_acc += get_accuracy(y_hat, y)
            train_n += 1

        train_ls_sum.append(train_ls / train_n)
        train_acc_sum.append(train_acc / train_n)

        test_n = 0
        model.eval()
        for X, y in test_iter:
            state = state.detach()

            y_hat, state = model(X, state)
            l = loss(y_hat, y)

            test_ls += l.item()
            test_acc += get_accuracy(y_hat, y)
            test_n += 1

        test_ls_sum.append(test_ls / test_n)
        test_acc_sum.append(test_acc / test_n)

        if (epoch + 1) % 20 == 0:
            print('epoch %d, train loss %f, train accuracy %f, test loss %f,'
                  ' test accuracy %f, sec %.2f' % (epoch + 1, train_ls / train_n,
                                                   train_acc / train_n, test_ls / test_n,
                                                   test_acc / test_n, time.time() - start))
    return np.array(train_ls_sum).mean(), np.array(train_acc_sum).mean(), np.array(test_ls_sum).mean(), np.array(test_acc_sum).mean()


def k_fold(k, X, y, num_epochs, lr, weight_decay, batch_size):
    """
    k折交叉验证
    :param k: int
            k折
    :param X: tensor
            shape: (num_seq, seq_length, w2v_dim)
            训练数据
    :param y: tensor
            shape: (num_seq, )
            训练样本
    :param num_epochs: int
            epoch的轮数
    :param lr: float
            学习率
    :param weight_decay: float
            权重衰减
    :param batch_size: int
            一个batch的大小
    :return:
    """
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X, y)
        model = RNN(128, 300, 2, False, 2)
        train_ls_sum, train_acc_sum, test_ls_sum, test_acc_sum = train(model,
                                                                       batch_size,
                                                                       X_train,
                                                                       y_train,
                                                                       X_valid,
                                                                       y_valid, lr,
                                                                       num_epochs,
                                                                       weight_decay)

        print('fold %d, avg train loss %f, avg train accuracy %f,'
              ' avg test loss %f, avg test accuracy %f' % (i, train_ls_sum,
                                                           train_acc_sum,
                                                           test_ls_sum,
                                                           test_acc_sum))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_w2v, labels = preprocess.preprocess()
X = torch.FloatTensor(X_w2v)
y = torch.LongTensor(labels)
k_fold(10, X, y, 100, 5, 0, 8)
