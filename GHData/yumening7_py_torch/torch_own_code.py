'''
@author: Dzh
@date: 2020/1/17 16:38
@file: torch_own_code.py
'''

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

print(train_data.data.size())
print(train_data.targets.size())
# plt.imshow(train_data.data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[2])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),              # 28*28像素大小的样本， 指定第一层输出为128
            nn.Tanh(),                          # 经过激活函数
            nn.Linear(128, 64),                 # 再次压缩
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)                    # 最后压缩到只有3个输出
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()                        # 由于训练集在每次训练的时候的取值范围会被压缩到0-1之间，所以这里也压缩
        )

    def forward(self, x):
        encoder = self.encoder(x)               # 自编码压缩
        decoder = self.decoder(encoder)         # 自编码解压
        return encoder, decoder

auto_encoder = AutoEncoder()
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
# 生成2行5列的子图，总画布大小为（5，2）英寸
fig, ax = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

# 选取指定样本的数据，转为二维矩阵，第一个维度保留batch_size，第二个维度是像素长宽的乘积
view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255

for i in range(N_TEST_IMG):
    # ax[0][i]代表用子图下标为0的行去便利绘图，np.reshape()传入两个数据，第一个是要reshape的数据，第二个是新定义的shape
    ax[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    ax[0][i].set_xticks(())         # 刻度为空
    ax[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28*28)             # 保留batch，然后数据大小定义为28*28
        b_y = x.view(-1, 28*28)             # 由于只是用到x的数据，所以这里用来对比的真实数据，也是x

        encoded, decoded = auto_encoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         if step % 100 == 0:
#             print('Epoch：', epoch, '| train loss：%.4f' % loss.data.numpy())
#
#             encoded_data, decoded_data = auto_encoder(view_data)
#             # print('decoded===', decoded_data.data.numpy())
#             for i in range(N_TEST_IMG):
#                 ax[1][i].clear()            # 清空子图下标为1的行对应列的图像
#                 # 用预测数据去子图下标1的行对应的列的绘图，用预测数据去绘图进行对比
#                 ax[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
#                 ax[1][i].set_xticks(())
#                 ax[1][i].set_yticks(())
#             plt.draw()
#             plt.pause(0.05)
# plt.ioff()
# plt.show()

view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255
encoded_data, decoded_data = auto_encoder(view_data)

print('type===', type(encoded_data))
fig = plt.figure(2)
ax = Axes3D(fig)
X = encoded_data[:, 0].detach().numpy()
Y = encoded_data[:, 1].numpy()
Z = encoded_data[:, 2].numpy()
values = train_data.data[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9));
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()

