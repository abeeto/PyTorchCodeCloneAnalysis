# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:35
# @Author : hhq
# @File : two_layer_torch.py
import torch
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)
# 暂时忽略偏置
learning_rate = 1e-6
for it in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    ##要将tensor转为单个数字
    loss = (y_pred - y).pow(2).sum().item()
    print(it, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
