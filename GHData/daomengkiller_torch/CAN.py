import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)  # 设置随机数种子

BATCH_SIZE = 64
LR_G = 0.0001  # 设置学习率
LR_D = 0.0001
N_IDEAS = 5  # 初始化激发数
ART_COMPONENTS = 15  # 生成的数据点
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])  # 为了进行批训练，多次产生点

plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')  # 上限
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')  # 下限
plt.legend(loc='upper right')
plt.show()


def artist_works():  #
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)


# 生成网络，输入激发数，输出画家的作品
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),

)
# 对抗网络，输入画家作品，输出概率
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
# 梯度优化函数
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()  # 产生一堆真实画家的作品
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))  # 矩阵激发数
    G_paintings = G(G_ideas)  # 伪造作品

    prob_artist0 = D(artist_paintings)  # 真实作品的判断
    prob_artist1 = D(G_paintings)  # 伪造作品的判断

    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))  # log函数，画图理解，loss尽量往零走
    G_loss = torch.mean(torch.log(1. - prob_artist1))  # 尽量使判断为假的概率为低

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True)  # retain_variables保留缓冲区的变量，以便后面使用
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
