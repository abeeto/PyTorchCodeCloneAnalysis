# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:31
# @Author : hhq
# @File : two_layers_np.py
import numpy as np

# N表示有多少输入，D_in表示每个输入有多少维，H表示隐藏层输出有多少维，D_out表示最后输出有多少维
N, D_in, H, D_out = 64, 1000, 100, 10
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
# 暂时忽略偏置

learning_rate = 1e-6
for it in range(500):
    ## 定义损失
    h = x.dot(w1)  # 矩阵相乘 64*100
    h_relu = np.maximum(h, 0)  # 用于逐元素比较两个array的大小。h_relu所有元素非负，用了relu函数，当函数值小于0时取0
    y_pred = h_relu.dot(w2)  # 64*10
    loss = np.square(y_pred - y).sum()
    print(it, loss)
    ## 求梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    ## 梯度下降
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
