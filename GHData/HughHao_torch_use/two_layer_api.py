# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:38
# @Author : hhq
# @File : two_layer_api.py
import torch
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)
learning_rate = 1e-6
for it in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    ## 要将tensor转为单个数字
    loss = (y_pred - y).pow(2).sum()
    print(it, loss.item())

    loss.backward()
    with torch.no_grad():  ##为了不让计算图占内存，就用torch.no_grad
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 根据pytorch中backward（）函数的计算，当网络参量进行反馈时，
        #  梯度是累积计算而不是被替换，但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，因此需
        #  要对每个batch调用一遍grad.zero_（）将参数梯度置0.
        w1.grad.zero_()
        w2.grad.zero_()
