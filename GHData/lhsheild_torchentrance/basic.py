import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 输入数据1*32*32
        # 第一层（卷积层）
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入1 输出6 卷积3*3
        # 第二层（卷积层）
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 第三层（全链接层）
        self.fc1 = nn.Linear(16 * 28 * 28, 512)  # 输入维度16*28*28=12544 输出维度512
        # 第四层（全链接层）
        self.fc2 = nn.Linear(512, 64)
        # 第五层（全链接层）
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):  # 定义数据流向
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1, 16 * 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


if __name__ == '__main__':
    # net = Net()
    # # 定义随机输入数据
    # input_data = torch.randn(1, 1, 32, 32)
    # print(input_data)
    # print(input_data.shape)
    # # 运行神经网络
    # result = net.forward(input_data)
    # print(f'result1: {result}')
    # print(f'result1_size: {result.size()}')
    # # 随机生成真实值
    # target = torch.randn(2)
    # target = target.view(1, -1)
    # print(f'target: {target}')
    # print(f'target_shape: {target.shape}')
    # # 计算损失
    # criterion = nn.L1Loss()  # 定义损失函数
    # loss = criterion(result, target)  # 计算损失
    # print(f'loss1: {loss}')
    # # 反向传递
    # net.zero_grad()  # 清零梯度
    # loss.backward()  # 自动计算梯度，反向传递
    # # 更新权重
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # optimizer.step()
    # # 重新计算
    # result = net.forward(input_data)
    # print(f'result2: {result}')
    # print(f'result2_size: {result.size()}')
    # loss = criterion(result, target)  # 计算损失
    # print(f'loss2: {loss}')

    # scalar
    x = tensor(42.)
    print(f'x: {x}')
    print(f'x: {x.dim()}')
    x = 2 * x
    print(f'x: {x}')
    print(f'x: {x.item()}')

    # vector
    v = tensor([1.5, 0.5, -0.5])
    print(f'v: {v}')
    print(f'v: {v.dim()}')
    v = 2 * v
    print(f'v: {v}')
    print(f'v: {v.size()}')

    # matrix
    m = tensor(
        [
            [1., 2.],
            [3., 4.]
        ]
    )
    print(f'm: {m}')
    m = 2 * m
    print(f'm: {m}')
    m = m.matmul(m)
    print(f'm: {m}')
