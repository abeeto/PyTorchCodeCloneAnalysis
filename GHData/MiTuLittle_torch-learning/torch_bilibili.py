#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
# author: hongtao.zhang
# date:2020-12-09

import numpy as np
import torch
from visdom import Visdom

NUM_DIGITS = 10


def fizz_buzz_encode(i_e):
    if i_e % 15 == 0:
        return 3
    elif i_e % 5 == 0:
        return 2
    elif i_e % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i_d, prediction):
    return [str(i_d), 'fizz', 'buzz', 'fizz buzz'][prediction]


def helper(i_h):
    fizz_res = fizz_buzz_decode(i_h, fizz_buzz_encode(i_h))
    return fizz_res


def binary_encode(i_b, num_digits):
    return np.array([i_b >> d & 1 for d in range(num_digits)][::-1])


class MyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        print('TEST Tensor')


tr_x = MyTensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
tr_y = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
print(tr_x.shape)
print(tr_y.shape)

NUM_HIDDEN = 100
model = torch.nn.Sequential(torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
                            torch.nn.ReLU(),
                            torch.nn.Linear(NUM_HIDDEN, 4))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

vis = Visdom(env='model-test')
BATCH_SIZE = 128
for epoch in range(500):
    for start in range(0, len(tr_x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_x = tr_x[start:end]
        batch_y = tr_y[start:end]

        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        print('Epoch', epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_x = MyTensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    test_y = model(test_x)
predicts = zip(range(1, 101), test_y.max(1)[1].cpu().data.tolist())
result_torch = [fizz_buzz_decode(i, x) for i, x in predicts]
i_c = 0
print(result_torch)
for i in range(1, 101):
    fizz_res_c = helper(i)
    print(fizz_res_c)
    print(result_torch[i - 1])
    if fizz_res_c == result_torch[i - 1]:
        i_c += 1
vis.line(Y=np.random.rand(10), opts=dict(showlegend=True))

Y = np.linspace(-5, 5, 100)
vis.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)
print(i_c, '正确个数')
