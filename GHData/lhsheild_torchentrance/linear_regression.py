# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/3 16:23
# @Author : ahrist
# @Email : 2693022425@qq.com
# @File : linear_regression.py
# @Software: PyCharm

import time
import numpy as np
import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


USE_GPU = True


if __name__ == '__main__':
    start = time.time()
    x_values = [i for i in range(11)]
    print(x_values)
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    print(x_train)
    print(x_train.shape)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    print(y_train)
    print(y_train.shape)

    linear_regression = LinearRegressionModel(1, 1)
    device = torch.device("cuda:0" if USE_GPU else "cpu")
    linear_regression.to(device)
    print(linear_regression)

    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(1000):
        epoch += 1
        if USE_GPU:
            inputs = torch.from_numpy(x_train).cuda()
            labels = torch.from_numpy(y_train).cuda()
        else:
            inputs = torch.from_numpy(x_train)
            labels = torch.from_numpy(y_train)
        optimizer.zero_grad()

        outputs = linear_regression(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    if USE_GPU:
        predicted = linear_regression(torch.from_numpy(x_train).cuda().requires_grad_()).cpu().data.numpy().argmax()
    else:
        predicted = linear_regression(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(predicted)

    torch.save(linear_regression.state_dict(), 'linear_regression.pt')
    end = time.time()
    print(end - start)
