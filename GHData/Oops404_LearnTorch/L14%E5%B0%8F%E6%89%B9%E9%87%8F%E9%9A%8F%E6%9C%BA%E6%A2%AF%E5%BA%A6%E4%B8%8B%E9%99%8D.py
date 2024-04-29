# -*- coding: UTF-8-*-
"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/22 15:26
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
from ipywidgets import interact, fixed

"""
梯度向量：
就是多元函数上，各个自变量的偏导数组成的向量，比如损失函数是L(w1,w2,b)，
在损失函数上对w1，w2，b三个自变量分别求偏导，求得的梯度向量表示就是：
[∂L/∂w1, ∂L/∂w2, ∂L/∂b]的转置。简写为grad L(w1,w2) 或者 ▽L(w1,w2)。

假设初始点(w_1(0),w_2(0)),梯度向量是(∂L/∂w_1, ∂L/∂w_2),那让坐标点
按照梯度向量的反方向移动的方法如下：
    w_1(1) = w_1(0) - ∂L / ∂w_1
    w_2(1) = w_2(0) - ∂L / ∂w_2
将两个w写在i同一个权重向量里，用t代表走到第t步(即进行第t次迭代)，则有：
    w_(t + 1) = w_t - ∂L / ∂w_t
加上步长η则：
    w_(t + 1) = w_t - η * (∂L / ∂w_t)
如上就是权重迭代公式。
"""

w1 = np.arange(-10, 10, 0.05)
w2 = np.arange(-10, 10, 0.05)
w1, w2 = np.meshgrid(w1, w2)
lossfn = (2 - w1 - w2) ** 2 + (4 - 3 * w1 - w2) ** 2


def plot_3D(elev=45, azim=60, X=w1, y=w2):
    """
    图解
    可以调到15度观察
    定义一个绘制三维图像的函数
    elev表示上下旋转的角度
    azim表示平行旋转的角度
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
    ax = plt.subplot(projection="3d")
    ax.plot_surface(w1, w2, lossfn, cmap='rainbow', alpha=0.7)
    ax.view_init(elev=elev, azim=azim)
    # ax.xticks([-10,-5,0,5,10])
    ax.set_xlabel("w1", fontsize=20)
    ax.set_ylabel("w2", fontsize=20)
    ax.set_zlabel("lossfn", fontsize=20)
    plt.show()


interact(plot_3D, elev=[0, 15, 30], azip=(-180, 180), X=fixed(w1), y=fixed(w2))
plt.show()
