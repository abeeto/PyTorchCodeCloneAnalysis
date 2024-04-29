import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def make_traindata():
    x = []
    y = []
    for i in range(4):
        x.append(i)
        y.append(i * 3 + 1)
    return x, y


def forward(x, w, b):
    return x * w + b


def loss_function(x_val, y_val, w, b):
    y_hat = forward(x_val, w, b)
    loss = (y_hat - y_val) ** 2
    return loss


def train(x, y):
    w, b = np.arange(0.0, 6.1, 0.1), np.arange(0.0, 2.1, 0.1)
    w, b = np.meshgrid(w, b)
    mse_list = []
    loss_mean = 0
    for x_val, y_val in zip(x, y):
        loss_mean += loss_function(x_val, y_val, w, b)
    loss_mean /= 4
    mse_list.append(loss_mean)
    show_plt3D(w, b, mse_list[0])


def show_plt3D(w, b, mse):
    # 定义figure
    fig = plt.figure(figsize=(10, 10), dpi=300)
    # 将figure变为3d
    ax = Axes3D(fig)
    # 绘图，rstride:行之间的跨度  cstride:列之间的跨度
    surf = ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # 设置z轴的长度从0到80.
    ax.set_zlim(0, 80)
    # 设置轴标签的图片标题
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("loss")
    ax.text(0.2, 2, 43, "Cost Value", color='black')
    # 设置一个把value映射到颜色的彩条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # 显示图片
    plt.show()


if __name__ == '__main__':
    x, y = make_traindata()
    train(x, y)
