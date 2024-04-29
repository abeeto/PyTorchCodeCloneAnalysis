# -*- coding: utf-8 -*-
# @Time : 2022/4/17 13:02
# @Author : hhq
# @File : gradient_show.py
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as pld


def f(x, y):
    return x ** 2 - y ** 2


def grad_x(f, x, y):
    h = 1e-4
    return (f(x + h / 2, y) - f(x - h / 2, y)) / h


def grad_y(f, x, y):
    h = 1e-4
    return (f(x, y + h / 2) - f(x, y - h / 2)) / h


def numerical_gradient(f, P):
    grad = np.zeros_like(P)
    for i in range(P[0].size):
        grad[0][i] = grad_x(f, P[0][i], P[1][i])
        grad[1][i] = grad_y(f, P[0][i], P[1][i])
    return grad


# 定义坐标轴
fig1 = plt.figure()
ax3 = plt.axes(projection='3d')
# ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

x = np.arange(-2, 2, 0.25)
y = np.arange(-2, 2, 0.25)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
# 作图
ax3.plot_surface(X, Y, Z, cmap='rainbow')
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值

X = X.flatten()
Y = Y.flatten()
fig2 = plt.figure()
grad = numerical_gradient(f, np.array([X, Y]))
plt.quiver(X, Y, grad[0], grad[1])  # grad 是一个1*X.size的数组
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
