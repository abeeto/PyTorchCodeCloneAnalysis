

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

# 超参数
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5             # 用多少个想法（随机数）去生成一个作品，也就是输入特征数（输入神经元）
ART_COMPONENTS = 15     # 一个作品最终对应的特征值数量，也是输出神经元
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])           # shape = (64, 15)
EPOCH = 10000

# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')              # x[0] 矩阵都进行了2倍平方再+1, 1 <= y <= 3
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')              # x[0] 矩阵都进行了平方, 0 <= y <= 1
# plt.legend(loc='upper right')
# plt.show()

# 艺术家的作品
def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]     # a.shape = (64, 1), 1 <= a < 2
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)               # paintings.shape = (64, 15), 0 <= paintings < 3 , paintings 的取值范围正好在上述绘制的两条线中间
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

# 生成器（新手画家）的神经网络，把想法转换为作品
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

# 判别器（新手鉴赏家）的神经网络，对作品进行评分，并转为好作品（艺术家作品）的概率
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

optim_G = torch.optim.Adam(G.parameters(), lr=LR_G)
optim_D = torch.optim.Adam(D.parameters(), lr=LR_D)

for step in range(EPOCH):
    artist_paintings = artist_works()                               # 艺术家生成了 BATCH_SIZE个作品
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))            # 用标准正态分布的方式生成一批随机数，对应的shape = (64, 5)
    G_paintings = G(G_ideas)                                        # 生成器（新手画家）也生成了 BATCH_SIZE个作品

    prob_artist0 = D(artist_paintings)                              # 用判别器（新手鉴赏家）判断艺术家的作品，prob_artist0是艺术家作品为艺术家作品的概率
    prob_artist1 = D(G_paintings)                                   # 用判别器（新手鉴赏家）判断生成器（新手画家）的作品，prob_artist1是生成器（新手画家）作品为艺术家作品的概率

    '''
    判别器（新手鉴赏家）的目的是为了区分出生成器（新手画家）和艺术家的作品
    最理想的情况下，prob_artist0中的所有值都为1，prob_artist1中的所有值都为0，此时的loss值为0。全部都判断正确。
    最不理想的情况下，prob_artist0中的所有值都为0，prob_artist1中的所有值都为1，此时的loss值为正无穷。全部都判断错误。
    如下公式可以很好的实现判别器的目的，所以用作损失函数。
    当prob_artist0越大，且prob_artist1越小时，判别器的loss值越小。
    '''
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1 - prob_artist1))
    '''
    生成器（新手画家）的目的是为了蒙混过关，希望欺骗过了判别器，让判别器认为自己的作品是艺术家作品。
    最理想的情况下，prob_artist1中的所有值都为1，此时的loss值为0，全部蒙混过关。
    最不理想的情况下。prob_artist1中的所有值都为0，此时的loss值为正无穷，没有一个蒙混过关。
    如下公式可以实现生成器（新手画家）的目的。
    当prob_artist1越大，生成器的loss越小。
    '''
    G_loss = -torch.mean(torch.log(prob_artist1))                   # 自己定义的loss
    # G_loss = torch.mean(torch.log(1 - prob_artist1))                  # 教程定义的loss

    # 接下来就是两个神经网络的误差反向传递
    optim_G.zero_grad()
    '''
    误差反向传递之后，保留下一些参数给之后的D_loss进行误差反向传递。
    默认retain_graph=False，默认状态下计算图内存反向传播之后会被释放掉，retain_graph=True是保留计算图内存。
    相当于判别器（新手鉴赏家）告诉生成器（新手画家）它是怎么判别艺术家的作品的。
    '''
    G_loss.backward(retain_graph=True)
    optim_G.step()

    optim_D.zero_grad()
    D_loss.backward()
    optim_D.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generator')                                # 从生成器（新手画家）的作品中选出一个来绘制
        plt.text(-0.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-0.5, 2, 'D score=%.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()

