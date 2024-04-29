import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 输入数据3*32*32
        # 第一层（卷积层）
        self.conv1 = nn.Conv2d(3, 6, 3)  # 输入1 输出6 卷积3*3
        # 第二层（卷积层）
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 第三层（全链接层）
        self.fc1 = nn.Linear(16 * 28 * 28, 512)  # 输入维度16*28*28=12544 输出维度512
        # 第四层（全链接层）
        self.fc2 = nn.Linear(512, 64)
        # 第五层（全链接层）
        self.fc3 = nn.Linear(64, 10)  # 输入纬度64 输出纬度10（数据集有十类）

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


def img_show(img):
    # torch.tensor [c, h, w]
    img = img / 2 + 0.5  # 返归一
    nping = img.numpy()
    nping = np.transpose(nping, (1, 2, 0))  # [h, w, c]
    plt.imshow(nping)
    plt.show()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ]
)

# 训练数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)

# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=8)

# 私人数据集

if __name__ == '__main__':
    # data_iter = iter(train_loader)  # 随机加载一个mini batch
    # images, labels = data_iter.next()
    # img_show(torchvision.utils.make_grid(images))

    # net = Net()
    # # 定义权值更新规则与损失函数
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), 0.001, momentum=0.9)
    # # 训练模型
    # for epoch in range(2):
    #     for i, data in enumerate(train_loader):
    #         images, labels = data
    #         output = net(images)
    #         loss = criterion(output, labels)
    #         # 更新权重
    #         optimizer.zero_grad()  # 清零梯度
    #         loss.backward()  # 自动计算梯度，反向传递
    #         optimizer.step()
    #
    #         if i % 250 == 0:
    #             print(f'Epoch: {epoch}, Step: {i}, Loss: {loss.item()}')
    #
    # # 测试模型
    # correct = 0.0
    # total = 0.0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         output = net(images)
    #         _, predicted = torch.max(output.data, 1)
    #         correct += (predicted == labels).sum()
    #         total += labels.size(0)
    # print(f'准确率：{float(correct/total)}')
    #
    # # 保存模型
    # torch.save(net.state_dict(), './models.pt')

    net = Net()
    net.load_state_dict(torch.load('./models.pt'))
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)
    print(f'准确率：{float(correct/total)}')